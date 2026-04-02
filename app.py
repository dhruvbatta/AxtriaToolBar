import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from openai import AzureOpenAI
import os
import json
import io
from io import BytesIO
from dotenv import load_dotenv
from typing import Dict
from datetime import datetime
import re


# Import news integration module
try:
    from news_integrated import fetch_drug_news, run_news_chatbot
    NEWS_MODULE_AVAILABLE = True
except ImportError:
    NEWS_MODULE_AVAILABLE = False
    print("News module not available. Event discovery from news will be disabled.")

# Import agentic functions (with fallback if not available)
try:
    from agentic_functions import apply_conversational_changes
except ImportError:
        def apply_conversational_changes(changes: Dict) -> bool:

        # """Enhanced fallback function for conversational parameter changes including custom metrics"""
            try:
                updated = False
                
                # Safe float conversion  
                def safe_float_convert(val, default=0.0):
                    if val is None:
                        return default
                    try:
                        return float(val)
                    except (ValueError, TypeError):
                        return default
                
                for change in changes.get('changes', []):
                    parameter = change.get('parameter')
                    new_value = change.get('new_value')
                    action = change.get('action', 'update')
                    
                    # Handle ADDING custom metrics
                    if action == 'add' and parameter == 'custom_metric':
                        if 'ai_assumptions_v2' in st.session_state:
                            ai = st.session_state['ai_assumptions_v2']
                            
                            # Extract metric details from new_value
                            metric_name = new_value.get('name', 'Custom Metric')
                            source_pop = new_value.get('source_population', 'Diagnosed Population')
                            rate = safe_float_convert(new_value.get('rate', 0.5))
                            
                            # Initialize derived_metrics if needed
                            if 'derived_metrics' not in ai:
                                ai['derived_metrics'] = {}
                            
                            # Create the rate and count metrics
                            rate_name = f"{metric_name} %"
                            count_name = f"{metric_name} Count"
                            
                            # Add rate metric
                            ai['derived_metrics'][rate_name] = {
                                'source': source_pop,
                                'base_rate': rate,
                                'growth_rate': 0.0,
                                'is_rate': True,
                                'growth_type': 'proportional'
                            }
                            
                            # Add count metric
                            ai['derived_metrics'][count_name] = {
                                'source': source_pop,
                                'rate_metric': rate_name,
                                'base_rate': rate,
                                'growth_rate': 0.0,
                                'is_rate': False,
                                'growth_type': 'proportional'
                            }
                            
                            # Add to metrics list
                            if 'metrics' not in ai:
                                ai['metrics'] = []
                            ai['metrics'].extend([rate_name, count_name])
                            
                            st.session_state['ai_assumptions_v2'] = ai
                            updated = True
                            st.success(f"✅ Added custom metric: {metric_name}")
                    
                    # Handle epidemiology parameter changes
                    elif parameter in ['prevalence', 'diagnosed', 'treated']:
                        if 'ai_assumptions_v2' in st.session_state:
                            ai = st.session_state['ai_assumptions_v2']
                            if 'known_values' not in ai:
                                ai['known_values'] = {}
                            ai['known_values'][parameter] = safe_float_convert(new_value)
                            st.session_state['ai_assumptions_v2'] = ai
                            updated = True
                    
                    # Handle brand parameter changes
                    elif parameter in ['peak_share', 'launch_year', 'adherence', 'price_per_dose', 'annual_dose']:
                        if 'brand_config' in st.session_state:
                            brand_config = st.session_state['brand_config']
                            if parameter in ['peak_share', 'adherence']:
                                brand_config[parameter] = safe_float_convert(new_value)
                            elif parameter in ['launch_year', 'annual_dose']:
                                brand_config[parameter] = int(new_value)
                            else:
                                brand_config[parameter] = new_value
                            st.session_state['brand_config'] = brand_config
                            updated = True
                    
                    # Handle custom metric modifications
                    elif action == 'update' and 'custom_metric' in parameter.lower():
                        if 'ai_assumptions_v2' in st.session_state:
                            ai = st.session_state['ai_assumptions_v2']
                            derived = ai.get('derived_metrics', {})
                            
                            # Find the metric to update
                            for metric_key in derived.keys():
                                if parameter.lower() in metric_key.lower():
                                    if 'rate' in metric_key.lower() or '%' in metric_key:
                                        derived[metric_key]['base_rate'] = safe_float_convert(new_value)
                                        updated = True
                                        break
                            
                            if updated:
                                st.session_state['ai_assumptions_v2'] = ai
                
                if updated:
                    st.session_state['current_forecast_state']['last_updated'] = datetime.now().isoformat()
                    # Force recalculation by clearing existing forecasts
                    if 'df_metrics_base' in st.session_state:
                        del st.session_state['df_metrics_base']
                    if 'df_metrics_custom' in st.session_state:
                        del st.session_state['df_metrics_custom']
                    if 'df_brand_forecast' in st.session_state:
                        del st.session_state['df_brand_forecast']
                
                return updated
            except Exception as e:
                st.error(f"Error applying conversational changes: {str(e)}")
                import traceback
                st.error(traceback.format_exc())
                return False

load_dotenv()

AZURE_OPENAI_ENDPOINT = os.getenv("AZURE_OPENAI_ENDPOINT")
DEPLOYMENT_CHAT = os.getenv("AZURE_DEPLOYMENT_CHAT")
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")

# reuse the same client pattern
client = AzureOpenAI(
    api_version="2024-02-01",
    azure_endpoint=AZURE_OPENAI_ENDPOINT,
    api_key=OPENAI_API_KEY
)

# ===== STREAMLIT PAGE CONFIG (MUST BE FIRST) =====
st.set_page_config(
    layout='wide',
    page_title='AI Epidemiology Forecast',
    page_icon='📊',
    initial_sidebar_state='expanded'
)

try:
    from formulas import ExcelModel
    HAS_FORMULAS = True
except ImportError:
    HAS_FORMULAS = False

# ===== EXAMPLE DATA CONSTANTS =====
EXAMPLE_HIV = """Disease: HIV
Geography: United States
Population: 341145670
Age Range: 25-60
Population Growth: 0.04% monthly
Prevalence: 14%"""

EXAMPLE_DIABETES = """Disease: Type 2 Diabetes
Geography: United States
Population: 341145670
Age Range: 40-70
Population Growth: 0.5% yearly
Prevalence: 10.5%
Diagnosed: 76%
Treated: 65%"""

EXAMPLE_CVD = """Disease: Cardiovascular Disease
Geography: United States
Population: 341145670
Age Range: 45-75
Population Growth: 0.6% yearly
Prevalence: 6.5%
Diagnosed: 80%
Treated: 70%"""

UPTAKE_CURVES = {
    '3 Year Slow': [0.10, 0.50, 1.00, 1.00, 1.00, 1.00, 1.00, 1.00, 1.00, 1.00, 1.00, 1.00, 1.00, 1.00, 1.00],
    '3 Year Fast': [0.60, 0.85, 1.00, 1.00, 1.00, 1.00, 1.00, 1.00, 1.00, 1.00, 1.00, 1.00, 1.00, 1.00, 1.00],
    '3 Year Medium': [0.26, 0.66, 1.00, 1.00, 1.00, 1.00, 1.00, 1.00, 1.00, 1.00, 1.00, 1.00, 1.00, 1.00, 1.00],
    '4 Year Slow': [0.08, 0.31, 0.69, 1.00, 1.00, 1.00, 1.00, 1.00, 1.00, 1.00, 1.00, 1.00, 1.00, 1.00, 1.00],
    '4 Year Fast': [0.33, 0.72, 0.88, 1.00, 1.00, 1.00, 1.00, 1.00, 1.00, 1.00, 1.00, 1.00, 1.00, 1.00, 1.00],
    '4 Year Medium': [0.20, 0.51, 0.79, 1.00, 1.00, 1.00, 1.00, 1.00, 1.00, 1.00, 1.00, 1.00, 1.00, 1.00, 1.00],
    '5 Year Slow': [0.06, 0.21, 0.50, 0.79, 1.00, 1.00, 1.00, 1.00, 1.00, 1.00, 1.00, 1.00, 1.00, 1.00, 1.00],
    '5 Year Fast': [0.28, 0.64, 0.82, 0.91, 1.00, 1.00, 1.00, 1.00, 1.00, 1.00, 1.00, 1.00, 1.00, 1.00, 1.00],
    '5 Year Medium': [0.17, 0.43, 0.66, 0.85, 1.00, 1.00, 1.00, 1.00, 1.00, 1.00, 1.00, 1.00, 1.00, 1.00, 1.00],
    '6 Year Slow': [0.04, 0.15, 0.36, 0.64, 0.85, 1.00, 1.00, 1.00, 1.00, 1.00, 1.00, 1.00, 1.00, 1.00, 1.00],
    '6 Year Fast': [0.24, 0.58, 0.76, 0.87, 0.93, 1.00, 1.00, 1.00, 1.00, 1.00, 1.00, 1.00, 1.00, 1.00, 1.00],
    '6 Year Medium': [0.15, 0.37, 0.56, 0.75, 0.89, 1.00, 1.00, 1.00, 1.00, 1.00, 1.00, 1.00, 1.00, 1.00, 1.00],
    '7 Year Slow': [0.05, 0.12, 0.27, 0.50, 0.73, 0.88, 1.00, 1.00, 1.00, 1.00, 1.00, 1.00, 1.00, 1.00, 1.00],
    '7 Year Fast': [0.21, 0.52, 0.71, 0.82, 0.89, 0.94, 1.00, 1.00, 1.00, 1.00, 1.00, 1.00, 1.00, 1.00, 1.00],
    '7 Year Medium': [0.13, 0.32, 0.49, 0.66, 0.81, 0.91, 1.00, 1.00, 1.00, 1.00, 1.00, 1.00, 1.00, 1.00, 1.00],
    '8 Year Slow': [0.05, 0.10, 0.22, 0.40, 0.61, 0.78, 0.90, 1.00, 1.00, 1.00, 1.00, 1.00, 1.00, 1.00, 1.00],
    '8 Year Fast': [0.19, 0.48, 0.66, 0.78, 0.86, 0.91, 0.94, 1.00, 1.00, 1.00, 1.00, 1.00, 1.00, 1.00, 1.00],
    '8 Year Medium': [0.12, 0.29, 0.44, 0.59, 0.73, 0.85, 0.92, 1.00, 1.00, 1.00, 1.00, 1.00, 1.00, 1.00, 1.00]
}


# Erosion curves (negative of uptake curves - for LoE modeling)
EROSION_CURVES = {}
for curve_name, curve_values in UPTAKE_CURVES.items():
    erosion_name = curve_name.replace('Year', 'Year Erosion')
    # Reverse the curve (1.0 down to the first value)
    EROSION_CURVES[erosion_name] = [1.0 - v + curve_values[0] for v in curve_values]
    
# ===== AGENTIC PARSING FUNCTIONS =====

# ===== ENHANCED AGENTIC DISCOVERY FUNCTION =====
def discover_disease_insights(disease_name: str, geography: str, include_funnel: bool = False, include_brand: bool = False) -> Dict:
    """AI agent that discovers segments, metrics, and brand targets based on disease knowledge."""
    
    system_prompt = """You are an expert epidemiologist and pharmaceutical market analyst.
    
Given a disease name and geography, provide clinical insights for patient segmentation and targeting.

RETURN ONLY valid JSON with this structure:
{
  "suggested_segments": [
    {
      "name": "Segment name",
      "rate": 0.XX,
      "source_population": "Diagnosed Population",
      "segment_type": "proportion",
      "growth_rate": 0.0,
      "rationale": "Why this segment matters clinically",
      "source_citation": "Author/Organization, Year, Study/Journal name, DOI or URL if available"
    }
  ],
  "suggested_linear_metrics": [
    {
      "name": "Metric name",
      "rate": 0.XX,
      "source_population": "Treated Population",
      "metric_type": "percentage",
      "rationale": "Clinical/commercial relevance for targeting",
      "source_citation": "Author/Organization, Year, Study/Journal name, DOI or URL if available"
    }
  ],
  "brand_insights": {
    "optimal_target": "Which population to target (e.g., 'High Risk', 'Treatment-eligible', 'Severe patients')",
    "typical_adherence": 0.XX,
    "typical_annual_dose": XXX,
    "typical_price_range": "$XX-$YY per dose",
    "typical_uptake": "X Year Medium",
    "rationale": "Why this is the best target for brand success"
  }
}

CRITICAL CITATION REQUIREMENTS:
✅ EVERY segment and metric MUST include a source_citation field
✅ Citations should reference:
   - Peer-reviewed medical literature
   - Clinical guidelines (e.g., ACC/AHA, ADA, GOLD)
   - Disease registries or epidemiological studies
   - Government health databases (CDC, WHO, Eurostat)
✅ Format: "Author/Org, Year, Study/Journal Name, DOI/URL"
✅ Examples:
   - "American Diabetes Association, 2024, Standards of Care, doi:10.2337/dc24-S001"
   - "WHO Global Tuberculosis Report, 2023, WHO Press"
   - "GOLD COPD Guidelines, 2024, goldcopd.org"

IMPORTANT GUIDELINES:

FOR SEGMENTS (when include_funnel=true):
- Provide 2-4 clinically meaningful patient groups that split a population
- Focus on: Disease severity (mild/moderate/severe), Risk levels (high/medium/low), Clinical phenotypes
- Segments should ADD UP to 100% of source population
- Use "Diagnosed Population" as source unless there's a strong clinical reason otherwise
- **MUST include source_citation with credible medical reference**
- Example for Diabetes: 
  {
    "name": "Uncontrolled Diabetes", 
    "rate": 0.35, 
    "source_population": "Diagnosed Population",
    "segment_type": "proportion",
    "growth_rate": 0.0,
    "rationale": "HbA1c >9%, at high risk for complications",
    "source_citation": "CDC National Diabetes Statistics Report, 2023, cdc.gov/diabetes/data"
  }

FOR LINEAR METRICS (when include_funnel=true):
- Identify 1-3 important targeting subsets (NOT segments - these are filters/subpopulations)
- Focus on: Treatment eligibility, High-risk criteria, Hospitalization history, Refractory cases
- These do NOT need to add up to 100% - they're independent filters
- **MUST include source_citation with credible medical reference**
- Example: 
  {
    "name": "Treatment-eligible", 
    "rate": 0.60, 
    "source_population": "Diagnosed Population",
    "metric_type": "percentage",
    "rationale": "Meets clinical criteria for pharmacotherapy",
    "source_citation": "ADA Standards of Care, 2024, Diabetes Care Journal"
  }

All rates as decimals (15% = 0.15, 60% = 0.60)
Base ALL suggestions on real clinical practice and therapeutic guidelines
DO NOT make up segments - use actual clinical classifications
NEVER omit source_citation - it is REQUIRED for every segment and metric"""

    # ... rest of function stays the same ...

    # Build conditional user prompt based on what's requested
    prompt_parts = [f"Disease: {disease_name}", f"Geography: {geography}", ""]
    
    if include_funnel:
        prompt_parts.append("""PROVIDE PATIENT SEGMENTATION INSIGHTS:

1. SUGGESTED SEGMENTS: 
   - Identify 2-4 clinically meaningful patient segments based on disease characteristics
   - Consider: Severity stages, risk stratification, clinical phenotypes, comorbidity patterns
   - Provide realistic percentages that total 100%
   - Each segment should have clinical relevance and treatment implications

2. SUGGESTED LINEAR METRICS:
   - Identify 1-3 key patient subsets important for treatment targeting
   - Consider: Eligibility criteria, risk factors, treatment history, disease complications
   - Provide realistic percentages (these don't need to sum to 100%)
   - Each metric should represent a commercially/clinically important population subset
   
Use actual clinical guidelines and disease staging systems where applicable.""")
    
    if include_brand:
        prompt_parts.append("""
3. BRAND TARGETING INSIGHTS:
   - Recommend optimal target population for a new branded therapy
   - Provide typical adherence, dosing, pricing, and uptake parameters for this disease area
   - Base recommendations on current treatment landscape and unmet needs""")
    
    prompt_parts.append("\nReturn ONLY the JSON structure. Omit sections not requested.")
    
    user_prompt = "\n".join(prompt_parts)

    try:
        response = client.chat.completions.create(
            model=DEPLOYMENT_CHAT,
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_prompt}
            ],
            temperature=0.3,
            max_tokens=2000
        )
        
        content = response.choices[0].message.content.strip()
        
        # Clean JSON
        if content.startswith('```json'):
            content = content[7:]
        if content.startswith('```'):
            content = content[3:]
        if content.endswith('```'):
            content = content[:-3]
        content = content.strip()
        
        discovered = json.loads(content)
        return discovered
        
    except Exception as e:
        st.error(f"Discovery error: {str(e)}")
        return {
            "suggested_segments": [],
            "suggested_linear_metrics": [],
            "brand_insights": {}
        }


def classify_disease_model_type(disease_name: str) -> Dict:
    """Use AI to classify if disease should use incidence-based or prevalence-based model.
    
    Returns:
        {
            "model_type": "incidence" or "prevalence",
            "rationale": "explanation",
            "suggested_incidence_rate": 0.XX (if incidence model),
            "suggested_prevalence": 0.XX (if prevalence model),
            "funnel_structure": [list of metric names]
        }
    """
    
    system_prompt = """You are an expert epidemiologist specializing in disease modeling approaches.

Classify whether a disease should use INCIDENCE-BASED or PREVALENCE-BASED forecasting model.

INCIDENCE-BASED DISEASES (New Cases Model):
✅ ALL Oncology/Cancer diseases (bladder, lung, breast, prostate, etc.)
✅ Acute conditions (MI, stroke, acute infections)
✅ Episodic diseases (seasonal flu, acute hepatitis)
✅ Diseases where "new cases per year" is the key metric
✅ Diseases with high mortality or cure rates (cases don't accumulate)

PREVALENCE-BASED DISEASES (Total Population Model):
✅ Chronic conditions (diabetes, hypertension, COPD)
✅ Lifelong conditions (HIV, autoimmune diseases)
✅ Progressive diseases with low mortality (Parkinson's, Alzheimer's)
✅ Diseases where "total living patients" is the key metric

RETURN ONLY valid JSON:
{
  "model_type": "incidence" or "prevalence",
  "rationale": "Brief explanation of why this model type applies",
  "suggested_incidence_rate": 0.XXXXX (annual new cases per 100,000 population - ONLY for incidence),
  "suggested_prevalence": 0.XX (total cases % - ONLY for prevalence),
  "funnel_structure": [
    "Metric 1 name",
    "Metric 2 name"
  ],
  "citation": "Medical source for rates"
}

FUNNEL STRUCTURES:

For INCIDENCE model (cancer):
[
  "Total Population",
  "Annual Incidence Rate",
  "Annual Incident Cases",
  "Diagnosed Rate",
  "Diagnosed Cases",
  "Treated Rate",
  "Treated Cases"
]

For PREVALENCE model (chronic):
[
  "Total Population",
  "Prevalence Rate",
  "Prevalent Population",
  "Diagnosed Rate",
  "Diagnosed Population",
  "Treated Rate",
  "Treated Population"
]

Use real epidemiological data and cite sources."""

    user_prompt = f"""Classify the forecasting model for: {disease_name}

Provide:
1. Model type (incidence or prevalence)
2. Rationale for this choice
3. Suggested rates with citations
4. Appropriate funnel structure

Return ONLY the JSON structure."""

    try:
        response = client.chat.completions.create(
            model=DEPLOYMENT_CHAT,
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_prompt}
            ],
            temperature=0,
            max_tokens=800
        )
        
        content = response.choices[0].message.content.strip()
        
        # Clean JSON
        if content.startswith('```json'):
            content = content[7:]
        if content.startswith('```'):
            content = content[3:]
        if content.endswith('```'):
            content = content[:-3]
        content = content.strip()
        
        classification = json.loads(content)
        return classification
        
    except Exception as e:
        st.warning(f"Disease classification error: {str(e)}. Defaulting to prevalence model.")
        # Default to prevalence model
        return {
            "model_type": "prevalence",
            "rationale": "Classification failed, defaulting to prevalence model",
            "suggested_incidence_rate": None,
            "suggested_prevalence": 0.05,
            "funnel_structure": [
                "Total Population",
                "Prevalence Rate",
                "Prevalent Population",
                "Diagnosed Rate",
                "Diagnosed Population",
                "Treated Rate",
                "Treated Population"
            ],
            "citation": "Default structure"
        }


# Complete the parse_input_agentic function implementation
def parse_input_agentic_core(text_input: str) -> Dict:
    """Core AI parsing logic with improved error handling."""
    system_prompt = """You are an expert epidemiology AI that extracts forecast parameters from natural language.

ANALYZE the user input and extract ALL relevant parameters.

Check for these keywords:
- "funnel" or "segments" or "granular" → User wants detailed patient segmentation
- "brand" or "drug" or "therapy" → User wants brand analysis
- Specific metrics mentioned → Extract those exactly

POPULATION AUTO-LOOKUP (CRITICAL):
[... keep existing population lookup section ...]

CRITICAL AGE SEGMENT DETECTION RULES:
✅ ALWAYS EXTRACT age segments when user mentions:
  - "Add age segments [range], [range]" → Extract each age range as a segment
  - "Age segments 20-45, 45-60, 60+" → Create segments for each range
  - "with age groups [ranges]" → Extract age groups
  - "split by age [ranges]" → Create age segments

✅ AGE SEGMENT FORMAT:
  - Name format: "Age [range]" (e.g., "Age 20-45", "Age 45-60", "Age 60+")
  - If rates NOT specified: Use equal distribution (e.g., 2 groups = 0.50 each, 3 groups = 0.33 each)
  - If rates ARE specified: Use those exact rates
  - Source population: Default to "Diagnosed Population"
  - Segment type: "proportion"

✅ EXAMPLES OF AGE SEGMENT EXTRACTION:

Input: "Add Age segments 20-45, 45-60 with funnel"
Output custom_segments: [
  {"name": "Age 20-45", "rate": 0.50, "source_population": "Diagnosed Population", "segment_type": "proportion", "growth_rate": 0.0},
  {"name": "Age 45-60", "rate": 0.50, "source_population": "Diagnosed Population", "segment_type": "proportion", "growth_rate": 0.0}
]

Input: "Diabetes forecast with age segments 18-44 (25%), 45-64 (50%), 65+ (25%)"
Output custom_segments: [
  {"name": "Age 18-44", "rate": 0.25, "source_population": "Diagnosed Population", "segment_type": "proportion", "growth_rate": 0.0},
  {"name": "Age 45-64", "rate": 0.50, "source_population": "Diagnosed Population", "segment_type": "proportion", "growth_rate": 0.0},
  {"name": "Age 65+", "rate": 0.25, "source_population": "Diagnosed Population", "segment_type": "proportion", "growth_rate": 0.0}
]

Input: "forecast with age groups 0-17, 18-64, 65+"
Output custom_segments: [
  {"name": "Age 0-17", "rate": 0.33, "source_population": "Diagnosed Population", "segment_type": "proportion", "growth_rate": 0.0},
  {"name": "Age 18-64", "rate": 0.33, "source_population": "Diagnosed Population", "segment_type": "proportion", "growth_rate": 0.0},
  {"name": "Age 65+", "rate": 0.34, "source_population": "Diagnosed Population", "segment_type": "proportion", "growth_rate": 0.0}
]

❌ DO NOT ignore age segments when explicitly mentioned
❌ DO NOT combine age segments with other segment types unless clearly separated

[... rest of existing system prompt ...]

CRITICAL SEGMENT DETECTION RULES:
✅ CREATE custom_segments when user says:
  - "with segments [name1] X%, [name2] Y%, [name3] Z%" (multiple segments splitting same source)
  - "segments: severe 30%, moderate 50%, mild 20%"
  - "split diagnosed into high/medium/low severity"
  - ANY mention of multiple named groups with percentages that ADD UP to 100%
  - **"Add age segments [range], [range]" → Create age-based segments**
  - **"with age groups [ranges]" → Create age-based segments**

❌ DO NOT create segments when:
  - Only ONE population subset is mentioned (use custom_linear_metrics instead)
  - Only disease name is mentioned without segment breakdown

[... keep rest of existing prompt ...]

FULL EXAMPLE WITH AGE SEGMENTS:

Input: "Forecast for Genital Herpes in United States from 2025-2035 Add Age segments 20-45, 45-60 with funnel"
Output: {
  "disease_name": "Genital Herpes",
  "geography": "United States",
  "population": 335000000,
  "age_range": null,
  "forecast_period": {
    "start_year": 2025,
    "end_year": 2035
  },
  "known_metrics": {
    "prevalence": null,
    "diagnosed": null,
    "treated": null
  },
  "request_funnel": true,
  "request_brand_analysis": false,
  "custom_segments": [
    {
      "name": "Age 20-45",
      "rate": 0.50,
      "source_population": "Diagnosed Population",
      "segment_type": "proportion",
      "growth_rate": 0.0
    },
    {
      "name": "Age 45-60",
      "rate": 0.50,
      "source_population": "Diagnosed Population",
      "segment_type": "proportion",
      "growth_rate": 0.0
    }
  ],
  "custom_linear_metrics": [],
  "brand_config": {},
  "market_events": [],
  "missing_parameters": [],
  "confidence": "high",
  "notes": "User requested age segmentation with funnel view"
}

REMEMBER:
- **When user says "Add Age segments X, Y" → ALWAYS create custom_segments with those age ranges**
- **Age segments should have "Age" prefix in name (e.g., "Age 20-45")**
- **If no rates given, distribute equally across age groups**
- **Age segments are custom_segments, not custom_linear_metrics**
- Percentages → decimals (15% → 0.15, 70% → 0.70, 80% → 0.80)
- Use null for unknown rates, NOT 0.0
- Single population subset → custom_linear_metrics
- Multiple segments adding to 100% → custom_segments
- Extract target_population from "peak share of X" or "targeting X"
- Extract ALL brand parameters when ANY brand info is mentioned
- ONLY return valid JSON, no explanatory text before or after
- Add any missing required parameters to missing_parameters list"""

    user_prompt = f"""Extract epidemiology forecast parameters from: {text_input}

Analyze carefully and extract:
1. Disease name and geography
2. Forecast period (start/end years)
3. ALL segment names with their percentages (if mentioned)
4. ALL brand parameters:
   - Brand name
   - Launch year
   - Peak share / market share
   - Price per dose
   - Annual dose / dosage
   - Adherence / compliance
   - Uptake curve type
   - Target population
5. Any custom metrics or events

IMPORTANT:
- If user mentions ANY brand information, populate brand_config with ALL mentioned fields
- Don't leave brand_config empty if brand is mentioned
- Convert all percentages to decimals

Return ONLY the JSON structure."""

    try:
        if not client:
            raise Exception("Azure OpenAI client not initialized")
            
        response = client.chat.completions.create(
            model=DEPLOYMENT_CHAT,
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_prompt}
            ],
            temperature=0.1,
            max_tokens=2000
        )
        
        result = response.choices[0].message.content.strip()
        
        # Debug: Show what AI returned
        st.write("🔍 AI Raw Response Preview:")
        st.code(result[:500] if len(result) > 500 else result, language="json")
        
        # Clean up the response - remove any non-JSON content
        if '```json' in result:
            start_idx = result.find('```json') + 7
            end_idx = result.find('```', start_idx)
            result = result[start_idx:end_idx].strip()
        elif '```' in result:
            start_idx = result.find('```') + 3
            end_idx = result.find('```', start_idx)
            result = result[start_idx:end_idx].strip()
        
        # Parse JSON response with better error handling
        import json
        try:
            parsed_data = json.loads(result)
            st.success(f"✅ Successfully parsed AI response")
            st.write(f"   - Found {len(parsed_data.get('custom_segments', []))} segments")
            st.write(f"   - Brand config populated: {bool(parsed_data.get('brand_config', {}).get('brand_name') or parsed_data.get('brand_config', {}).get('peak_share'))}")
        except json.JSONDecodeError as e:
            st.error(f"JSON Parse Error: {str(e)}")
            st.error(f"Raw AI Response: {result[:500]}...")
            # Try to extract JSON from response
            start_idx = result.find('{')
            end_idx = result.rfind('}')
            if start_idx >= 0 and end_idx > start_idx:
                json_str = result[start_idx:end_idx+1]
                try:
                    parsed_data = json.loads(json_str)
                except json.JSONDecodeError:
                    raise Exception(f"Unable to parse AI response as JSON: {str(e)}")
            else:
                raise Exception("No valid JSON found in AI response")
        
        # Validate and set defaults
        if not isinstance(parsed_data, dict):
            raise Exception("AI response is not a valid dictionary")
        
        # Ensure all required fields exist with proper defaults
        parsed_data.setdefault('disease_name', None)
        parsed_data.setdefault('geography', None)
        parsed_data.setdefault('population', None)
        parsed_data.setdefault('age_range', None)
        parsed_data.setdefault('forecast_period', {'start_year': 2025, 'end_year': 2035})
        parsed_data.setdefault('known_metrics', {})
        parsed_data.setdefault('custom_segments', [])
        parsed_data.setdefault('custom_linear_metrics', [])
        parsed_data.setdefault('brand_config', {})
        parsed_data.setdefault('market_events', [])
        parsed_data.setdefault('missing_parameters', [])
        parsed_data.setdefault('confidence', 'medium')
        parsed_data.setdefault('notes', '')
        
        # Debug output for brand config
        brand_cfg = parsed_data.get('brand_config', {})
        if brand_cfg:
            st.write("**Brand Configuration Extracted:**")
            for key, value in brand_cfg.items():
                if value is not None:
                    st.write(f"   - {key}: {value}")
        
        return parsed_data
        
    except Exception as e:
        st.error(f"Error parsing input with AI: {str(e)}")
        raise  # Re-raise to trigger fallback

def parse_input_fallback_robust(text_input: str) -> Dict:
    """Ultra-robust fallback parser that extracts basic info from natural language."""
    result = {
        'disease_name': None,
        'geography': None,
        'population': None,
        'age_range': None,
        'forecast_period': {'start_year': 2025, 'end_year': 2035},
        'known_metrics': {},
        'custom_segments': [],
        'custom_linear_metrics': [],
        'brand_config': {},
        'market_events': [],
        'missing_parameters': [],
        'confidence': 'low',
        'notes': text_input
    }
    
    text = text_input.lower()
    
    # Extract disease
    diseases = ['hypertension', 'diabetes', 'cardiovascular', 'heart failure', 'hiv', 'cancer', 'copd', 'asthma', 'obesity']
    for disease in diseases:
        if disease in text:
            result['disease_name'] = disease.title()
            break
    
    # Extract geography
    if 'usa' in text or 'united states' in text or 'us ' in text:
        result['geography'] = 'United States'
    elif 'europe' in text:
        result['geography'] = 'Europe'
    elif 'global' in text:
        result['geography'] = 'Global'
    
    # Extract years for forecasting
    import re
    years = re.findall(r'\b(20\d{2})\b', text_input)
    if len(years) >= 2:
        result['forecast_period'] = {
            'start_year': int(min(years)),
            'end_year': int(max(years))
        }
    
    # Extract brand info
    if 'brand' in text or 'launch' in text:
        brand_match = re.search(r'(\d+)%?\s*peak\s*share', text)
        launch_match = re.search(r'launch\w*\s+(\d{4})', text)
        
        if brand_match:
            peak_share = float(brand_match.group(1)) / 100
            launch_year = int(launch_match.group(1)) if launch_match else (int(years[0]) if years else 2025)
            
            result['brand_config'] = {
                'brand_name': 'New Brand',
                'launch_year': launch_year,
                'peak_share': peak_share,
                'target_population': 'Treated Population',
                'price_per_dose': None,
                'annual_dose': None
            }
    
    # STRICTER SEGMENT DETECTION - Only create segments if EXPLICITLY mentioned with specific patterns
    segment_patterns = [
        r'segments?:\s*([^,\n]+(?:,\s*[^,\n]+)*)',  # "segments: high, medium, low"
        r'create\s+segments?[^\n]*?([^,\n]+(?:,\s*[^,\n]+)*)',  # "create segments high, medium, low"
        r'segments?\s+([^,\n]+(?:,\s*[^,\n]+)*)',  # "segments high, medium, low"
    ]
    
    segments_found = False
    for pattern in segment_patterns:
        match = re.search(pattern, text, re.IGNORECASE)
        if match:
            segments_text = match.group(1)
            # Check if it contains multiple named segments
            if ('high' in segments_text.lower() and 'medium' in segments_text.lower() and 'low' in segments_text.lower()) or \
               ('severe' in segments_text.lower() and 'moderate' in segments_text.lower() and 'mild' in segments_text.lower()) or \
               ('class i' in segments_text.lower() and 'class ii' in segments_text.lower()):
                segments_found = True
                break
    
    # Only create segments if explicitly found with clear patterns
    if segments_found:
        if 'class i' in text and 'class ii' in text:
            # Obesity classifications
            result['custom_segments'] = [
                {
                    'name': 'Class III Obesity',
                    'source_population': 'Diagnosed Population',
                    'rate': 0.15,
                    'segment_type': 'proportion',
                    'growth_rate': 0.0
                },
                {
                    'name': 'Class II Obesity',
                    'source_population': 'Diagnosed Population',
                    'rate': 0.25,
                    'segment_type': 'proportion',
                    'growth_rate': 0.0
                },
                {
                    'name': 'Class I Obesity',
                    'source_population': 'Diagnosed Population',
                    'rate': 0.60,
                    'segment_type': 'proportion',
                    'growth_rate': 0.0
                }
            ]
        else:
            # Generic severity levels
            result['custom_segments'] = [
                {
                    'name': 'High Severity',
                    'source_population': 'Diagnosed Population',
                    'rate': 0.30,
                    'segment_type': 'proportion',
                    'growth_rate': 0.0
                },
                {
                    'name': 'Medium Severity', 
                    'source_population': 'Diagnosed Population',
                    'rate': 0.50,
                    'segment_type': 'proportion',
                    'growth_rate': 0.0
                },
                {
                    'name': 'Low Severity',
                    'source_population': 'Diagnosed Population',
                    'rate': 0.20,
                    'segment_type': 'proportion',
                    'growth_rate': 0.0
                }
            ]
    
    # Set missing parameters
    missing = []
    if not result['disease_name']:
        missing.append('disease_name')
    if not result['geography']:
        missing.append('geography')
    if not result['population']:
        missing.append('population')
    
    result['missing_parameters'] = missing
    return result

# Update the main parse function to use proper fallback chain
def parse_input_agentic(text_input: str) -> Dict:
    """Advanced AI-powered natural language parser with robust fallbacks."""
    try:
        # Try the AI parser first
        return parse_input_agentic_core(text_input)
    except Exception as e:
        st.warning(f"AI parser failed: {str(e)}. Using fallback parser.")
        # Try the simple parser
        try:
            simple_result = parse_input_v2_fallback(text_input)
            # Convert simple result to new format
            return {
                'disease_name': simple_result.get('disease_name'),
                'geography': simple_result.get('geography'),
                'population': simple_result.get('population'),
                'age_range': simple_result.get('age_range'),
                'forecast_period': {'start_year': 2025, 'end_year': 2035},
                'known_metrics': simple_result.get('known_metrics', {}),
                'custom_segments': [],
                'custom_linear_metrics': [],
                'brand_config': {},
                'market_events': [],
                'missing_parameters': [],
                'confidence': 'medium',
                'notes': simple_result.get('notes', text_input)
            }
        except Exception as e2:
            st.warning(f"Simple parser also failed: {str(e2)}. Using robust fallback.")
            # Use the ultra-robust fallback
            return parse_input_fallback_robust(text_input)

def parse_input_v2_fallback(text_input: str) -> Dict:
    """Legacy parser for backward compatibility with safe float conversion."""
    result = {
        'disease_name': None,
        'geography': None,
        'population': None,
        'age_range': None,
        'population_growth': None,
        'known_metrics': {},
        'notes': ''
    }

    lines = [l.strip() for l in text_input.split('\n') if l.strip()]
    notes = []
    
    def safe_float_parse(value_str):
        """Safely parse float from string, return None if invalid"""
        if not value_str:
            return None
        try:
            # Remove common formatting
            clean_str = value_str.replace(',', '').replace('%', '').strip()
            val = float(clean_str)
            # If original had %, convert to decimal
            if '%' in value_str:
                val = val / 100.0
            return val
        except (ValueError, TypeError):
            return None
    
    for line in lines:
        if ':' not in line:
            notes.append(line)
            continue
        
        key, value = [p.strip() for p in line.split(':', 1)]
        key_lower = key.lower()
        
        if 'disease' in key_lower:
            result['disease_name'] = value
        elif 'geography' in key_lower or 'country' in key_lower or 'region' in key_lower:
            result['geography'] = value
        elif 'population' in key_lower and 'growth' not in key_lower:
            pop_val = safe_float_parse(value)
            if pop_val is not None:
                result['population'] = int(pop_val)
        elif 'age' in key_lower:
            result['age_range'] = value
        elif 'growth' in key_lower:
            # Handle population growth
            growth_val = safe_float_parse(value)
            if growth_val is not None:
                freq = 'yearly'
                if 'month' in value.lower():
                    freq = 'monthly'
                result['population_growth'] = {'rate': growth_val, 'freq': freq}
        else:
            # Try to parse as known metric
            parsed_val = safe_float_parse(value)
            if parsed_val is not None:
                if 'prevalence' in key_lower or 'preval' in key_lower:
                    result['known_metrics']['prevalence'] = parsed_val
                elif 'diagnosed' in key_lower or 'diagnos' in key_lower:
                    result['known_metrics']['diagnosed'] = parsed_val
                elif 'treated' in key_lower or 'treatment' in key_lower:
                    result['known_metrics']['treated'] = parsed_val
                elif 'incidence' in key_lower or 'incid' in key_lower:
                    result['known_metrics']['incidence'] = parsed_val
                else:
                    notes.append(line)
            else:
                notes.append(line)

    result['notes'] = '\n'.join(notes)
    return result

# ===== CONVERSATION MANAGEMENT FUNCTIONS =====

def init_conversation_state():
    """Initialize conversation state variables."""
    if 'conversation_history' not in st.session_state:
        st.session_state['conversation_history'] = []
    if 'agentic_mode' not in st.session_state:
        st.session_state['agentic_mode'] = True
    if 'pending_clarifications' not in st.session_state:
        st.session_state['pending_clarifications'] = []
    if 'current_forecast_state' not in st.session_state:
        st.session_state['current_forecast_state'] = {}
    if 'conversation_context' not in st.session_state:
        st.session_state['conversation_context'] = {
            'original_prompt': None,
            'extracted_parameters': None,
            'forecast_complete': False,
            'last_action': None
        }

def add_conversation_turn(user_input: str, assistant_response: str, action_type: str = 'general'):
    """Add a conversation turn to history."""
    turn = {
        'timestamp': datetime.now().strftime('%H:%M:%S'),
        'user_input': user_input,
        'assistant_response': assistant_response,
        'action_type': action_type,  # 'initial_request', 'clarification', 'adjustment', 'general'
        'forecast_state_snapshot': st.session_state.get('current_forecast_state', {}).copy()
    }
    st.session_state['conversation_history'].append(turn)
    st.session_state['conversation_context']['last_action'] = action_type

def detect_parameter_changes(user_input: str) -> Dict:
    """Detect if user wants to modify existing parameters or add custom metrics."""
    system_prompt = """You are analyzing user input for parameter modification requests.

DETECT if the user wants to:
1. Change existing forecast parameters
2. Add new custom metrics
3. Modify brand parameters
4. Add events

RETURN ONLY valid JSON (no additional text):
{
  "has_changes": boolean,
  "change_type": "modify|add|remove",
  "parameter_category": "epidemiology|brand|custom_metric|events",
  "changes": [
    {
      "parameter": "parameter_name",
      "action": "add|update|remove",
      "new_value": value_or_object
    }
  ],
  "requires_recalculation": boolean
}

EXAMPLES:

"increase adherence to 75%" -> {
  "has_changes": true,
  "change_type": "modify",
  "parameter_category": "brand",
  "changes": [{"parameter": "adherence", "action": "update", "new_value": 0.75}]
}

"add custom linear metric after diagnosed population called eligible population and rate is 80%" -> {
  "has_changes": true,
  "change_type": "add",
  "parameter_category": "custom_metric",
  "changes": [{
    "parameter": "custom_metric",
    "action": "add",
    "new_value": {
      "name": "Eligible Population",
      "source_population": "Diagnosed Population",
      "rate": 0.80,
      "metric_type": "percentage"
    }
  }]
}

"change severe cases to 25%" -> {
  "has_changes": true,
  "change_type": "modify",
  "parameter_category": "custom_metric",
  "changes": [{"parameter": "severe_cases", "action": "update", "new_value": 0.25}]
}"""

    try:
        response = client.chat.completions.create(
            model=DEPLOYMENT_CHAT,
            messages=[
                {'role': 'system', 'content': system_prompt},
                {'role': 'user', 'content': f"Analyze: {user_input}"}
            ],
            temperature=0,
            max_tokens=1000
        )
        
        result = response.choices[0].message.content.strip()
        
        # Clean up response
        if '```json' in result:
            start_idx = result.find('```json') + 7
            end_idx = result.find('```', start_idx)
            result = result[start_idx:end_idx].strip()
        elif '```' in result:
            start_idx = result.find('```') + 3
            end_idx = result.find('```', start_idx)
            result = result[start_idx:end_idx].strip()
        
        # Parse JSON response
        import json
        try:
            parsed_result = json.loads(result)
        except json.JSONDecodeError as e:
            st.error(f"JSON Parse Error: {str(e)}")
            st.error(f"Raw Response: {result[:500]}")
            # Try to extract JSON
            start_idx = result.find('{')
            end_idx = result.rfind('}')
            if start_idx >= 0 and end_idx > start_idx:
                json_str = result[start_idx:end_idx+1]
                try:
                    parsed_result = json.loads(json_str)
                except json.JSONDecodeError:
                    return {'has_changes': False, 'changes': []}
            else:
                return {'has_changes': False, 'changes': []}
                
        return parsed_result
    except Exception as e:
        st.error(f"Error detecting changes: {str(e)}")
        return {'has_changes': False, 'changes': []}

def generate_clarifying_questions(missing_params: list, context: Dict) -> list:
    """Generate intelligent clarifying questions for missing parameters."""
    system_prompt = """Generate 1-3 specific clarifying questions for missing epidemiology parameters.

Context:
- Disease: {disease}
- Geography: {geography}
- Missing: {missing}

RETURN ONLY valid JSON array (no additional text):
[
  {{
    "question": "What is the total population size for {disease} patients in {geography}?",
    "parameter": "population",
    "suggested_default": "50000000",
    "explanation": "This affects all downstream calculations"
  }}
]

Make questions specific, actionable, and disease-appropriate.""".format(
        disease=context.get('disease_name', 'the specified disease'),
        geography=context.get('geography', 'the target geography'),
        missing=missing_params
    )

    try:
        response = client.chat.completions.create(
            model=DEPLOYMENT_CHAT,
            messages=[
                {'role': 'system', 'content': system_prompt},
                {'role': 'user', 'content': f"Generate questions for missing: {missing_params}"}
            ],
            temperature=0.3,
            max_tokens=800
        )
        
        result = response.choices[0].message.content.strip()
        
        # Clean up response
        if '```json' in result:
            start_idx = result.find('```json') + 7
            end_idx = result.find('```', start_idx)
            result = result[start_idx:end_idx].strip()
        
        # Parse JSON response with better error handling
        import json
        try:
            parsed_result = json.loads(result)
        except json.JSONDecodeError as e:
            st.error(f"JSON Parse Error in questions: {str(e)}")
            st.error(f"Raw Response: {result}")
            # Try to extract JSON array
            start_idx = result.find('[')
            end_idx = result.rfind(']')
            if start_idx >= 0 and end_idx > start_idx:
                json_str = result[start_idx:end_idx+1]
                try:
                    parsed_result = json.loads(json_str)
                except json.JSONDecodeError:
                    return [{
                        "question": f"Please provide: {', '.join(missing_params)}",
                        "parameter": "general",
                        "suggested_default": None,
                        "explanation": "This information is needed for the forecast"
                    }]
            else:
                return [{
                    "question": f"Please provide: {', '.join(missing_params)}",
                    "parameter": "general", 
                    "suggested_default": None,
                    "explanation": "This information is needed for the forecast"
                }]
                
        return parsed_result if isinstance(parsed_result, list) else [parsed_result]
    except Exception as e:
        st.error(f"Error generating clarifying questions: {str(e)}")
        return [{
            "question": f"Please provide more information about: {', '.join(missing_params)}",
            "parameter": "general",
            "suggested_default": None,
            "explanation": "This information is needed to complete the forecast"
        }]

# ===== SCENARIO MANAGEMENT FUNCTIONS =====

def save_scenario(scenario_name: str, description: str = ""):
    """Save current forecast state as a scenario"""
    if 'saved_scenarios' not in st.session_state:
        st.session_state['saved_scenarios'] = {}
    
    scenario_data = {
    'name': scenario_name,
    'description': description,
    'timestamp': datetime.now().strftime('%Y-%m-%d %H:%M:%S'),
    'epi_forecast': st.session_state.get('df_metrics_custom') if 'df_metrics_custom' in st.session_state else st.session_state.get('df_metrics_base'),
    'brand_forecast': st.session_state.get('df_brand_forecast'),
    'brand_config': st.session_state.get('brand_config'),
    'ai_assumptions': st.session_state.get('ai_assumptions_v2'),
    'parsed_input': st.session_state.get('parsed'),
    'frequency': st.session_state.get('frequency'),
    'start_year': st.session_state.get('start_year'),
    'end_year': st.session_state.get('end_year'),
    'events': st.session_state.get('brand_events', [])  # Add events
}
    
    st.session_state['saved_scenarios'][scenario_name] = scenario_data
    return True

def load_scenario(scenario_name: str):
    """Load a saved scenario into current session"""
    if 'saved_scenarios' not in st.session_state:
        return False
    
    if scenario_name not in st.session_state['saved_scenarios']:
        return False
    
    scenario = st.session_state['saved_scenarios'][scenario_name]
    
    # Restore all state
    if scenario['epi_forecast'] is not None:
        if 'df_metrics_custom' in st.session_state:
            st.session_state['df_metrics_custom'] = scenario['epi_forecast']
        else:
            st.session_state['df_metrics_base'] = scenario['epi_forecast']
    
    if scenario['brand_forecast'] is not None:
        st.session_state['df_brand_forecast'] = scenario['brand_forecast']
    
    if scenario['brand_config'] is not None:
        st.session_state['brand_config'] = scenario['brand_config']
    
    if scenario['ai_assumptions'] is not None:
        st.session_state['ai_assumptions_v2'] = scenario['ai_assumptions']
    
    if scenario['parsed_input'] is not None:
        st.session_state['parsed'] = scenario['parsed_input']
    
    return True

def compare_scenarios(scenario_names: list):
    """Generate comparison data for selected scenarios"""
    if 'saved_scenarios' not in st.session_state:
        return None
    
    comparison_data = {
        'scenarios': [],
        'configs': [],
        'revenue_data': {},
        'patient_data': {},
        'share_data': {}
    }
    
    for name in scenario_names:
        if name not in st.session_state['saved_scenarios']:
            continue
        
        scenario = st.session_state['saved_scenarios'][name]
        brand_forecast = scenario['brand_forecast']
        brand_config = scenario['brand_config']
        
        if brand_forecast is None or brand_config is None:
            continue
        
        comparison_data['scenarios'].append(name)
        comparison_data['configs'].append(brand_config)
        
        # Extract key metrics
        years = [str(col) for col in brand_forecast.columns]
        comparison_data['revenue_data'][name] = brand_forecast.loc['Revenue in billions'].tolist()
        comparison_data['patient_data'][name] = brand_forecast.loc['Patients on Brand X'].tolist()
        comparison_data['share_data'][name] = brand_forecast.loc['Brand X Patient Share'].tolist()
    
    if comparison_data['scenarios']:
        comparison_data['years'] = years
    
    return comparison_data


def orchestrate_agentic_forecast(user_prompt: str) -> Dict:
    """Main agentic workflow orchestrator with robust error handling."""
    try:
        # Step 1: Parse user input with AI
        with st.spinner("🤖 Analyzing your request..."):
            parsed_data = parse_input_agentic_core(user_prompt)
        
        # Store original prompt and parsed data
        st.session_state['conversation_context']['original_prompt'] = user_prompt
        st.session_state['conversation_context']['extracted_parameters'] = parsed_data
        
        # NEW STEP 1.5: Classify disease model type (incidence vs prevalence)
        disease_classification = None
        disease_name = parsed_data.get('disease_name')
        
        if disease_name:
            with st.spinner(f"🔬 Determining forecasting approach for {disease_name}..."):
                disease_classification = classify_disease_model_type(disease_name)
                
                model_type = disease_classification['model_type'].upper()
                st.success(f"**Model Type Detected**: {model_type}-BASED")
                st.info(f"📋 {disease_classification['rationale']}")
                
                # Display citation directly without nested expander
                if disease_classification.get('citation'):
                    st.caption(f"📚 Source: {disease_classification['citation']}")
                
                # Store classification in session state
                st.session_state['disease_classification'] = disease_classification
        
        # Step 1.5: AUTO-DISCOVERY - If user requests funnel or minimal info provided
        request_funnel = parsed_data.get('request_funnel', False)
        request_brand = parsed_data.get('request_brand_analysis', False)
        
        # Check if we should auto-discover (minimal user input)
        has_minimal_segments = len(parsed_data.get('custom_segments', [])) == 0
        has_minimal_metrics = len(parsed_data.get('custom_linear_metrics', [])) == 0
        has_disease = parsed_data.get('disease_name') is not None
        has_geography = parsed_data.get('geography') is not None
        
        if has_disease and (request_funnel or request_brand):
            st.info("🔍 **Auto-Discovery Mode:** Researching optimal segments, metrics, and targeting strategies...")
            
            with st.spinner("🧠 Discovering disease-specific insights..."):
                discovered = discover_disease_insights(
                    disease_name=parsed_data.get('disease_name'),
                    geography=parsed_data.get('geography', 'United States'),
                    include_funnel=request_funnel,
                    include_brand=request_brand
                )
            
            # Apply discovered segments if user requested funnel and hasn't specified segments
            # Apply discovered segments if user requested funnel and hasn't specified segments
            if request_funnel and has_minimal_segments and discovered.get('suggested_segments'):
                st.write("### 📊 Suggested Patient Segments:")
                
                # Get or initialize citations dictionary
                if 'ai_assumptions_v2' in st.session_state:
                    ai = st.session_state['ai_assumptions_v2']
                    if 'citations' not in ai:
                        ai['citations'] = {}
                else:
                    ai = {'citations': {}}
                
                for seg in discovered['suggested_segments']:
                    st.write(f"- **{seg['name']}**: {seg['rate']*100:.1f}% of {seg['source_population']}")
                    st.caption(f"   _{seg.get('rationale', '')}_")
                    
                    # Store citation for this segment
                    citation = seg.get('source_citation', 'AI-generated suggestion - no specific citation provided')
                    ai['citations'][seg['name']] = citation
                
                # Update session state with citations
                st.session_state['ai_assumptions_v2'] = ai
                
                # Add to parsed_data
                parsed_data['custom_segments'] = discovered['suggested_segments']
                st.success(f"✅ Added {len(discovered['suggested_segments'])} suggested segments with citations")
            
            # Apply discovered linear metrics if user requested funnel
            if request_funnel and has_minimal_metrics and discovered.get('suggested_linear_metrics'):
                st.write("### 🎯 Suggested Targeting Metrics:")
                
                # Get or initialize citations dictionary
                if 'ai_assumptions_v2' in st.session_state:
                    ai = st.session_state['ai_assumptions_v2']
                    if 'citations' not in ai:
                        ai['citations'] = {}
                else:
                    ai = {'citations': {}}
                
                for metric in discovered['suggested_linear_metrics']:
                    st.write(f"- **{metric['name']}**: {metric['rate']*100:.1f}% of {metric['source_population']}")
                    st.caption(f"   _{metric.get('rationale', '')}_")
                    
                    # Store citation for this metric
                    citation = metric.get('source_citation', 'AI-generated suggestion - no specific citation provided')
                    ai['citations'][metric['name']] = citation
                
                # Update session state with citations
                st.session_state['ai_assumptions_v2'] = ai
                
                # Add to parsed_data
                parsed_data['custom_linear_metrics'] = discovered['suggested_linear_metrics']
                st.success(f"✅ Added {len(discovered['suggested_linear_metrics'])} targeting metrics with citations")
            
            # Apply brand insights if user requested brand analysis
            if request_brand and discovered.get('brand_insights'):
                insights = discovered['brand_insights']
                brand_config = parsed_data.get('brand_config', {})
                
                st.write("### 💊 Brand Targeting Insights:")
                st.write(f"**Optimal Target:** {insights.get('optimal_target', 'Treated Population')}")
                st.write(f"**Typical Adherence:** {insights.get('typical_adherence', 0.68)*100:.0f}%")
                st.write(f"**Typical Dosing:** {insights.get('typical_annual_dose', 360)} doses/year")
                st.write(f"**Price Range:** {insights.get('typical_price_range', '$10-$20 per dose')}")
                st.write(f"**Uptake Curve:** {insights.get('typical_uptake', '7 Year Medium')}")
                st.caption(f"_{insights.get('rationale', '')}_")
                
                # Auto-fill brand config with discovered insights if not specified
                if not brand_config.get('target_population'):
                    brand_config['target_population'] = insights.get('optimal_target', 'Treated Population')
                
                if not brand_config.get('adherence'):
                    brand_config['adherence'] = insights.get('typical_adherence', 0.68)
                
                if not brand_config.get('annual_dose'):
                    brand_config['annual_dose'] = insights.get('typical_annual_dose', 360)
                
                if not brand_config.get('uptake_type'):
                    brand_config['uptake_type'] = insights.get('typical_uptake', '7 Year Medium')
                
                parsed_data['brand_config'] = brand_config
                st.success("✅ Applied brand targeting insights")
        
        # Step 2: Check for missing critical parameters
        missing_params = parsed_data.get('missing_parameters', [])
        confidence = parsed_data.get('confidence', 'medium')
        
        if missing_params and confidence == 'low':
            questions = generate_clarifying_questions(missing_params, parsed_data)
            st.session_state['pending_clarifications'] = questions
            
            add_conversation_turn(
                user_prompt,
                f"I need clarification on {len(questions)} parameters to generate an accurate forecast.",
                'clarification'
            )
            return {'status': 'needs_clarification', 'questions': questions, 'parsed_data': parsed_data}
        
        # Step 3: Auto-generate epidemiology assumptions
        with st.spinner("🔬 Generating epidemiology assumptions..."):
            population_val = parsed_data.get('population')
            if population_val is None:
                if parsed_data.get('geography') == 'United States':
                    population_val = 341000000
                elif parsed_data.get('geography') == 'Europe':
                    population_val = 750000000
                else:
                    population_val = 50000000
            
            input_data = {
                'disease_name': parsed_data.get('disease_name'),
                'geography': parsed_data.get('geography'),
                'population': population_val,
                'age_range': parsed_data.get('age_range'),
                'known_metrics': parsed_data.get('known_metrics', {}),
                'notes': f"Agentic forecast request: {user_prompt}",
                'disease_classification': disease_classification  # Pass classification to forecast agent
            }
            
            ai_assumptions = build_ai_forecast_agent_v2(input_data)
            
        # Step 3.5: Extract age segments from AI assumptions if present
        age_segments_from_ai = ai_assumptions.get('age_segments', [])
        if age_segments_from_ai:
            st.success(f"✅ AI found {len(age_segments_from_ai)} age distribution segments")
            for seg in age_segments_from_ai:
                st.write(f"   - {seg['name']}: {seg['rate']*100:.1f}%")
        
        # Step 4: Add custom segments and linear metrics if specified
        custom_segments = parsed_data.get('custom_segments', [])
        custom_linear_metrics = parsed_data.get('custom_linear_metrics', [])
        
        # PRIORITY: Add age segments FIRST if they exist from AI
        if age_segments_from_ai:
            # Insert age segments at the beginning of custom_segments
            custom_segments = age_segments_from_ai + custom_segments
            st.info(f"📊 Age segments will be placed first after Diagnosed {ai_assumptions.get('model_type', 'Population').title()}")
        
        st.write(f"🔍 DEBUG - Found {len(custom_segments)} total segments (including age splits)")
        st.write(f"🔍 DEBUG - Found {len(custom_linear_metrics)} custom linear metrics")
        
        if custom_segments or custom_linear_metrics:
            ai_assumptions.setdefault('derived_metrics', {})
            
            # Process custom segments with validation
            for idx, segment in enumerate(custom_segments):
                try:
                    # Validate segment structure
                    if not isinstance(segment, dict):
                        st.warning(f"Skipping invalid segment #{idx+1}: not a dictionary")
                        continue
                    
                    segment_name = segment.get('name')
                    if not segment_name:
                        st.warning(f"Skipping segment #{idx+1}: missing 'name' field")
                        continue
                    
                    # CRITICAL FIX: Strip any % signs from the name to avoid double % in metric names
                    segment_name = segment_name.replace('%', '').strip()
                    
                    segment_rate_name = f"{segment_name} %"
                    segment_count_name = f"{segment_name} Count"
                    segment_type = segment.get('segment_type', 'proportion')
                    segment_rate = segment.get('rate', 0.1)
                    segment_growth = segment.get('growth_rate', 0.0)
                    source_pop = segment.get('source_population', 'Diagnosed Population')
                    
                    st.write(f"✅ Adding segment: {segment_name} ({segment_rate*100:.1f}% of {source_pop})")
                    
                    # Add rate metric
                    ai_assumptions['derived_metrics'][segment_rate_name] = {
                        'source': source_pop,
                        'base_rate': float(segment_rate),
                        'growth_rate': float(segment_growth) if segment_type == 'linear_growth' else 0.0,
                        'is_rate': True,
                        'type': 'percentage',
                        'growth_type': 'proportional'
                    }
                    
                    # Add count metric
                    ai_assumptions['derived_metrics'][segment_count_name] = {
                        'source': source_pop,
                        'rate_metric': segment_rate_name,
                        'base_rate': float(segment_rate),
                        'growth_rate': float(segment_growth) if segment_type == 'linear_growth' else 0.0,
                        'is_rate': False,
                        'type': 'count',
                        'growth_type': 'proportional'
                    }
                    
                    # Add to metrics list RIGHT AFTER the source population COUNT metric (not rate)
                    # CRITICAL: Skip rate metrics (%, Rate) - only insert after Count/Population metrics
                    source_variations = [
                        source_pop,  # e.g., "Diagnosed Population"
                        source_pop.replace(' Population', ''),  # e.g., "Diagnosed" (but not "Diagnosed Rate")
                        source_pop.replace('Prevalent', 'Prevalence'),  # Handle Prevalent -> Prevalence
                        source_pop.replace('Treated', 'Treatment')  # Handle Treated -> Treatment
                    ]
                    
                    # Find the last occurrence of the source population COUNT metric (not rate)
                    insert_idx = len(ai_assumptions['metrics'])  # Default to end if not found
                    for i in range(len(ai_assumptions['metrics']) - 1, -1, -1):  # Search backwards
                        metric = ai_assumptions['metrics'][i]
                        # Skip if this is a rate metric (contains % or ends with 'Rate')
                        if '%' in metric or metric.endswith('Rate') or metric.endswith('%'):
                            continue
                        # Check if this is our source population count metric
                        for var in source_variations:
                            if var and var.lower() in metric.lower():
                                insert_idx = i + 1
                                break
                        if insert_idx < len(ai_assumptions['metrics']):
                            break

                    if segment_rate_name not in ai_assumptions['metrics']:
                        ai_assumptions['metrics'].insert(insert_idx, segment_rate_name)
                        insert_idx += 1
                    if segment_count_name not in ai_assumptions['metrics']:
                        ai_assumptions['metrics'].insert(insert_idx, segment_count_name)
                    
                except Exception as e:
                    st.error(f"Error processing segment '{segment.get('name', f'#{idx+1}')}': {str(e)}")
                    continue
            
            # Process custom linear metrics with validation - UPDATED
            for idx, linear_metric in enumerate(custom_linear_metrics):
                try:
                    # Validate linear metric structure
                    if not isinstance(linear_metric, dict):
                        st.warning(f"Skipping invalid linear metric #{idx+1}: not a dictionary")
                        continue
                    
                    metric_name = linear_metric.get('name')
                    if not metric_name:
                        st.warning(f"Skipping linear metric #{idx+1}: missing 'name' field")
                        continue
                    
                    # CRITICAL FIX: Strip any % signs from the name to avoid double % in metric names
                    metric_name = metric_name.replace('%', '').strip()
                    
                    # Get the rate - this is the KEY FIX
                    rate = linear_metric.get('rate', 0.0)
                    
                    # CRITICAL: Check if rate is 0.0 and warn
                    if rate == 0.0:
                        st.error(f"⚠️ Linear metric '{metric_name}' has rate 0.0 - this will produce 0 counts!")
                        st.error(f"   Check AI parsing - it should extract the percentage from your prompt")
                        st.write(f"   Linear metric data: {linear_metric}")
                        continue
                    
                    metric_type = linear_metric.get('metric_type', 'percentage')
                    source_pop = linear_metric.get('source_population', 'Treated Population')
                    growth_rate = linear_metric.get('growth_rate', 0.0)
                    
                    st.write(f"✅ Adding custom linear metric: {metric_name} at {rate*100:.1f}% of {source_pop}")
                    
                    # Add percentage metric
                    ai_assumptions['derived_metrics'][f"{metric_name} %"] = {
                        'source': source_pop,
                        'base_rate': float(rate),
                        'growth_rate': float(growth_rate),
                        'is_rate': True,
                        'type': 'percentage',
                        'growth_type': 'proportional'
                    }
                    
                    # Add count metric
                    ai_assumptions['derived_metrics'][f"{metric_name} Count"] = {
                        'source': source_pop,
                        'rate_metric': f"{metric_name} %",  # Link to the rate metric
                        'base_rate': float(rate),
                        'growth_rate': float(growth_rate),
                        'is_rate': False,
                        'type': 'count',
                        'growth_type': 'proportional'
                    }
                    
                    # Add to metrics list RIGHT AFTER the source population COUNT metric (not rate)
                    # CRITICAL: Skip rate metrics (%, Rate) - only insert after Count/Population metrics
                    source_population_variations = [
                        source_pop,  # e.g., "Diagnosed Population"
                        source_pop.replace(' Population', ''),  # e.g., "Diagnosed" (but not rate)
                        source_pop.replace('Prevalent', 'Prevalence'),  # Handle Prevalent -> Prevalence
                        source_pop.replace('Treated', 'Treatment')  # Handle Treated -> Treatment
                    ]
                    
                    # Find the last occurrence of the source population COUNT metric (not rate)
                    insert_index = len(ai_assumptions['metrics'])  # Default to end if not found
                    for i in range(len(ai_assumptions['metrics']) - 1, -1, -1):  # Search backwards
                        existing_metric = ai_assumptions['metrics'][i]
                        # Skip if this is a rate metric (contains % or ends with 'Rate')
                        if '%' in existing_metric or existing_metric.endswith('Rate') or existing_metric.endswith('%'):
                            continue
                        # Check if this is our source population count metric
                        for variation in source_population_variations:
                            if variation and variation.lower() in existing_metric.lower():
                                insert_index = i + 1
                                break
                        if insert_index < len(ai_assumptions['metrics']):
                            break
                    
                    # Insert the new metrics at the calculated position
                    if f"{metric_name} %" not in ai_assumptions['metrics']:
                        ai_assumptions['metrics'].insert(insert_index, f"{metric_name} %")
                        insert_index += 1
                    if f"{metric_name} Count" not in ai_assumptions['metrics']:
                        ai_assumptions['metrics'].insert(insert_index, f"{metric_name} Count")
                    
                    st.success(f"   → Created '{metric_name} %' and '{metric_name} Count' after '{source_pop}'")
                        
                except Exception as e:
                    st.error(f"Error processing linear metric '{linear_metric.get('name', f'#{idx+1}')}': {str(e)}")
                    import traceback
                    st.error(traceback.format_exc())
                    continue
        
        # Step 5: Generate epidemiology forecast
        with st.spinner("Generating epidemiology forecast..."):
            forecast_period = parsed_data.get('forecast_period', {})
            start_year = forecast_period.get('start_year', 2025)
            end_year = forecast_period.get('end_year', 2035)
            
            df_metrics = generate_forecast_v2(ai_assumptions, start_year, end_year, 'Yearly')
            
        # Step 6: Generate brand forecast if brand config provided
        df_brand = None
        brand_config = parsed_data.get('brand_config', {})
        
        st.write(f"DEBUG - Brand config from parsing: {brand_config}")
        
        # Check if brand_config has meaningful data
        has_brand = (
            brand_config and 
            isinstance(brand_config, dict) and 
            (brand_config.get('brand_name') or brand_config.get('peak_share', 0) > 0)
        )
        
        if has_brand:
            with st.spinner("Generating brand forecast..."):
                # Safe conversion helpers
                def safe_float(value, default):
                    if value is None:
                        return default
                    try:
                        return float(value)
                    except (ValueError, TypeError):
                        return default
                
                def safe_int(value, default):
                    if value is None:
                        return default
                    try:
                        return int(value)
                    except (ValueError, TypeError):
                        return default
                
                # Build complete brand config with safe conversions
                target_population = brand_config.get('target_population', 'Treated Population')
                target_population = brand_config.get('target_population', 'Treated Population')

# ADD THIS LINE - Safety check for None
                if target_population is None or not target_population:
                    target_population = 'Treated Population'
                
                # CRITICAL FIX: Normalize target population name to match metric names
                # Strip any % from target population name
                if target_population:
                    target_population = target_population.replace('%', '').strip()
                
                # If custom linear metrics exist but brand didn't specify target, use the custom metric
                if custom_linear_metrics and len(custom_linear_metrics) > 0:
                    # If target is still default "Treated Population", check if we should use custom metric instead
                    if target_population in ['Treated Population', 'Treated patients', 'Treated']:
                        # Use the first custom linear metric as the target
                        first_metric = custom_linear_metrics[0]
                        metric_name = first_metric.get('name', '').replace('%', '').strip()
                        if metric_name:
                            target_population = f"{metric_name} Count"
                            st.success(f"Auto-detected: Brand will target custom metric '{target_population}' (instead of default Treated Population)")
                    # If target is explicitly set to a custom metric name, normalize it
                    elif target_population and not target_population.endswith('Population') and not target_population.endswith('Count'):
                        for metric in custom_linear_metrics:
                            metric_name = metric.get('name', '').replace('%', '').strip()
                            if metric_name and target_population and (metric_name.lower() in target_population.lower() or target_population.lower() in metric_name.lower()):
                                # Use the count metric name
                                target_population = f"{metric_name} Count"
                                st.write(f"Brand will target custom metric: {target_population}")
                                break
                
                # Build brand_config_complete BEFORE using it in news search
                brand_config_complete = {
                    'brand_name': brand_config.get('brand_name', 'New Brand'),
                    'launch_year': safe_int(brand_config.get('launch_year'), start_year + 2),
                    'peak_share': safe_float(brand_config.get('peak_share'), 0.20),
                    'uptake_type': brand_config.get('uptake_type', '7 Year Medium'),
                    'target_population': target_population,
                    'adherence': safe_float(brand_config.get('adherence'), 0.68),
                    'annual_dose': safe_int(brand_config.get('annual_dose'), 360),
                    'price_per_dose': safe_float(brand_config.get('price_per_dose'), 16.0)
                }
                
                # Step 6.5: Extract events from news if brand name mentioned
                market_events_from_news = []
                brand_name_search = brand_config_complete.get('brand_name', '')
                
                # if brand_name_search and brand_name_search != 'New Brand' and NEWS_MODULE_AVAILABLE:
                #     st.write(f"🔍 Searching for market events related to: {brand_name_search}")
                #     try:
                #         with st.spinner(f"Fetching news for {brand_name_search}..."):
                #             news_results = fetch_drug_news(brand_name_search, n_results=10, filter_relevance=False)
                            
                #         if news_results:
                #             st.write(f"✓ Found {len(news_results)} news articles")
                            
                #             # Convert news to events
                #             for news_item in news_results:
                #                 # Only include news with negative impact or competitive intelligence
                #                 impact = news_item.get('impact') or 'Neutral'
                #                 category = news_item.get('category') or ''
                                
                #                 # Safety check for None values
                #                 if category and 'competitor' in category.lower() or impact in ['Negative', 'Mostly Negative']:
                #                     event_year = None
                #                     pub_date = news_item.get('publication_date', '')
                #                     if pub_date:
                #                         try:
                #                             event_year = int(pub_date.split('-')[0])
                #                         except:
                #                             pass
                                    
                #                     if event_year and start_year <= event_year <= end_year:
                #                         market_events_from_news.append({
                #                             'event_year': event_year,
                #                             'event_type': 'competitor_entry',
                #                             'event_name': news_item.get('title', 'Market Event'),
                #                             'event_description': news_item.get('snippet', ''),
                #                             'post_event_peak_share': brand_config_complete['peak_share'] * 0.85,  # 15% reduction
                #                             'years_to_peak': 3,
                #                             'curve_type': '3 Year Medium Erosion',
                #                             'curve_category': 'Erosion (Decline)',
                #                             'description': f"{news_item.get('title', 'Event')}: {news_item.get('snippet', '')[:100]}"
                #                         })
                            
                #             if market_events_from_news:
                #                 st.success(f"✅ Found {len(market_events_from_news)} relevant market events from news")
                #                 market_events.extend(market_events_from_news)
                #     except Exception as e:
                #         st.warning(f"Could not fetch news events: {str(e)}")
                
                st.write(f"✅ Generating brand forecast for: {brand_config_complete['brand_name']}")
                st.write(f"   → Launch Year: {brand_config_complete['launch_year']}")
                st.write(f"   → Peak Share: {brand_config_complete['peak_share']*100:.1f}%")
                st.write(f"   → Target Population: {brand_config_complete['target_population']}")
                st.write(f"   → Uptake Curve: {brand_config_complete['uptake_type']}")
                st.write(f"   → Adherence: {brand_config_complete['adherence']*100:.1f}%")
                
                # Convert market events to proper format
                market_events = []
                for event in parsed_data.get('market_events', []):
                    if event.get('event_type') == 'competitor_entry':
                        market_events.append({
                            'event_year': event.get('event_year', start_year + 5),
                            'event_type': 'competitor_entry',
                            'post_event_peak_share': brand_config_complete['peak_share'] - event.get('impact_magnitude', 0.10),
                            'years_to_peak': event.get('impact_duration', 5),
                            'curve_type': '5 Year Medium Erosion',
                            'curve_category': 'Erosion (Decline)',
                            'description': event.get('description', 'Competitor entry')
                        })
                
                df_brand = generate_brand_forecast(
                    df_metrics, brand_config_complete, start_year, end_year, market_events
                )
                
                # Store complete config
                brand_config = brand_config_complete
        
        # Step 7: Store results in session state
        st.session_state['df_metrics_base'] = df_metrics
        st.session_state['ai_assumptions_v2'] = ai_assumptions
        st.session_state['parsed'] = input_data
        st.session_state['frequency'] = 'Yearly'
        st.session_state['start_year'] = start_year
        st.session_state['end_year'] = end_year
        
        if df_brand is not None:
            st.session_state['df_brand_forecast'] = df_brand
            st.session_state['brand_config'] = brand_config
            st.session_state['brand_events'] = market_events
        
        st.session_state['current_forecast_state'] = {
            'epidemiology_complete': True,
            'brand_complete': df_brand is not None,
            'has_custom_segments': len(custom_segments) > 0,
            'has_linear_metrics': len(custom_linear_metrics) > 0,
            'has_news_events': len(market_events_from_news) > 0 if 'market_events_from_news' in locals() else False,
            'last_updated': datetime.now().isoformat()
        }
        
        st.session_state['conversation_context']['forecast_complete'] = True
        
        add_conversation_turn(
            user_prompt,
            "Forecast generated successfully! You can now view results, make adjustments, or ask questions.",
            'initial_request'
        )
        
        return {
            'status': 'success',
            'epidemiology_forecast': df_metrics,
            'brand_forecast': df_brand,
            'ai_assumptions': ai_assumptions,
            'brand_config': brand_config,
            'custom_segments': custom_segments,
            'custom_linear_metrics': custom_linear_metrics
        }
        
    except Exception as e:
        st.error(f"Error in agentic forecast: {str(e)}")
        import traceback
        st.error("Full error trace:")
        st.code(traceback.format_exc())
        add_conversation_turn(
            user_prompt,
            f"Sorry, I encountered an error: {str(e)}. Please try rephrasing your request.",
            'error'
        )
        return {'status': 'error', 'message': str(e)}


def parse_input_v2(text_input: str) -> Dict:
    """Parse the user's structured input into a dictionary of parameters.

    Expected simple key:value lines. Recognizes:
      Disease, Geography, Population, Age Range, Population Growth
      Incidence, Prevalence, Diagnosed, Treated
    Returns normalized dict with defaults.
    """
    result = {
        'disease_name': None,
        'geography': None,
        'population': None,
        'age_range': None,
        'population_growth': None,  # dict {'rate':float,'freq':'monthly'/'yearly'}
        'known_metrics': {},
        'notes': ''
    }

    lines = [l.strip() for l in text_input.split('\n') if l.strip()]
    notes = []
    for line in lines:
        if ':' in line:
            key, val = [p.strip() for p in line.split(':', 1)]
            k = key.lower().replace(' ', '_')
            if k in ('disease', 'disease_name'):
                result['disease_name'] = val
            elif k in ('geography', 'region', 'country'):
                result['geography'] = val
            elif k == 'population':
                try:
                    result['population'] = int(val.replace(',', ''))
                except Exception:
                    notes.append(f"population parse failed: {val}")
            elif k in ('age_range', 'age-range', 'age'):
                result['age_range'] = val
            elif 'growth' in k or 'population_growth' in k:
                # example: 0.04% monthly or 0.5% yearly or 0.0004 monthly
                txt = val.lower()
                rate = None
                try:
                    # extract number
                    parsed = ''.join(c for c in txt if (c.isdigit() or c == '.' or c == '%'))
                    if '%' in txt:
                        parsed = parsed.replace('%', '')
                        rate = float(parsed) / 100.0
                    else:
                        rate = float(parsed)
                except Exception:
                    rate = None
                freq = 'monthly' if 'month' in txt else 'yearly'
                if rate is not None:
                    result['population_growth'] = {'rate': rate, 'freq': freq}
                else:
                    notes.append(f"could not parse population growth: {val}")
            elif k in ('incidence', 'prevalence', 'diagnosed', 'treated'):
                try:
                    num = float(''.join(c for c in val if (c.isdigit() or c == '.' or c == '%')))
                    if '%' in val:
                        num = num / 100.0
                    result['known_metrics'][k] = num
                except Exception:
                    notes.append(f"could not parse metric {key}: {val}")
            else:
                notes.append(line)
        else:
            notes.append(line)

    result['notes'] = '\n'.join(notes)
    return result


def parse_input_v2_fallback(text_input: str) -> Dict:
    """Legacy parser for backward compatibility."""
    result = {
        'disease_name': None,
        'geography': None,
        'population': None,
        'age_range': None,
        'population_growth': None,
        'known_metrics': {},
        'notes': ''
    }

    lines = [l.strip() for l in text_input.split('\n') if l.strip()]
    notes = []
    for line in lines:
        if ':' in line:
            key, value = line.split(':', 1)
            key = key.strip().lower()
            value = value.strip()
            
            if 'disease' in key:
                result['disease_name'] = value
            elif 'geography' in key or 'country' in key or 'region' in key:
                result['geography'] = value
            elif 'population' in key and 'growth' not in key:
                try:
                    result['population'] = int(value.replace(',', '').replace(' ', ''))
                except:
                    notes.append(f"Could not parse population: {value}")
            elif 'age' in key:
                result['age_range'] = value
            elif 'growth' in key:
                try:
                    if '%' in value:
                        result['population_growth'] = float(value.replace('%', '')) / 100
                    else:
                        result['population_growth'] = float(value)
                except:
                    notes.append(f"Could not parse growth rate: {value}")
            elif any(x in key for x in ['prevalence', 'incidence']):
                try:
                    if '%' in value:
                        result['known_metrics']['prevalence'] = float(value.replace('%', '')) / 100
                    else:
                        result['known_metrics']['prevalence'] = float(value)
                except:
                    notes.append(f"Could not parse prevalence: {value}")
            elif 'diagnosed' in key:
                try:
                    if '%' in value:
                        result['known_metrics']['diagnosed'] = float(value.replace('%', '')) / 100
                    else:
                        result['known_metrics']['diagnosed'] = float(value)
                except:
                    notes.append(f"Could not parse diagnosed rate: {value}")
            elif 'treated' in key:
                try:
                    if '%' in value:
                        result['known_metrics']['treated'] = float(value.replace('%', '')) / 100
                    else:
                        result['known_metrics']['treated'] = float(value)
                except:
                    notes.append(f"Could not parse treated rate: {value}")
            else:
                notes.append(line)
        else:
            notes.append(line)

    result['notes'] = '\n'.join(notes)
    return result

def validate_citation(citation: str) -> tuple:
    """Basic validation for citations"""
    if not citation or len(citation.strip()) < 10:
        return False, "Citation is too short. Please provide author, year, and source."
    
    # Check for year (4 digits)
    if not re.search(r'\b(19|20)\d{2}\b', citation):
        return False, "Citation should include a year (e.g., 2023)"
    
    # Check for some common citation elements
    has_author = any(word[0].isupper() for word in citation.split() if len(word) > 2)
    
    if not has_author:
        return False, "Citation should include author name(s)"
    
    return True, "Valid citation"


def build_ai_forecast_agent_v2(input_data: Dict) -> Dict:
    """Call AI to infer default assumptions with research citations."""
    
    def safe_population_convert(value, default=1000000):
        """Safely convert population value to integer, handling None and invalid values"""
        if value is None:
            return default
        try:
            return int(float(value))
        except (ValueError, TypeError):
            return default
    
    # Safe population extraction
    safe_population = safe_population_convert(input_data.get('population'))
    
    defaults = {
        'metrics': ['Total Population', 'Prevalence Rate', 'Prevalent Population', 'Diagnosed Rate', 'Diagnosed Population', 'Treated Rate', 'Treated Population'],
        'base_year': datetime.now().year,
        'growth_rates': {'Population': 0.007},
        'population': safe_population,
        'known_values': input_data.get('known_metrics', {}),
        'assumptions': 'Default assumptions used.',
        'citations': {}  # NEW: Store citations
    }

    # Normalize known_values helper
    def _normalize_known_values(raw: Dict) -> Dict:
        out = {}

        def _extract_number(s):
            """Return a float for strings like '0.3% (0.003)...' or '0.003' or '12%' or numeric types.
            Returns None if no number found."""
            if s is None:
                return None
            if isinstance(s, (int, float)):
                return float(s)
            txt = str(s)
            # Prefer percent form if present (e.g. '0.3% (...)')
            pct_match = re.search(r'([-+]?\d*\.?\d+)\s*%', txt)
            if pct_match:
                try:
                    return float(pct_match.group(1)) / 100.0
                except Exception:
                    pass
            # otherwise find first plain number
            num_match = re.search(r'([-+]?\d*\.?\d+)', txt)
            if num_match:
                try:
                    return float(num_match.group(1))
                except Exception:
                    pass
            return None

        for k, v in (raw or {}).items():
            kn = k.lower().replace('_', ' ').strip()
            parsed = _extract_number(v)
            # Only use parsed value if it's not None, otherwise skip this key
            if parsed is not None:
                val = parsed
            else:
                # Skip None values - don't include them in the output
                continue

            if 'preval' in kn:
                out['prevalence'] = val
            elif 'diagnos' in kn:
                out['diagnosed'] = val
            elif 'treat' in kn:
                out['treated'] = val
            elif 'incid' in kn:
                out['incidence'] = val
            else:
                out[kn.replace(' ','_')] = val
        
        return out

    # Normalize defaults known_values coming from user input
    defaults['known_values'] = _normalize_known_values(defaults.get('known_values', {}))
    
    # Check if disease classification was provided to determine model type
    disease_classification = input_data.get('disease_classification')
    is_incidence_model = False
    
    if disease_classification and disease_classification.get('model_type') == 'incidence':
        is_incidence_model = True
        # Update default metrics for incidence model
        defaults['metrics'] = [
            'Total Population',
            'Annual Incidence Rate',
            'Annual Incident Cases',
            'Diagnosed Rate',
            'Diagnosed Cases',
            'Treated Rate',
            'Treated Cases'
        ]
        defaults['model_type'] = 'incidence'
        # Add suggested incidence rate if provided
        if disease_classification.get('suggested_incidence_rate'):
            defaults['known_values']['incidence'] = disease_classification['suggested_incidence_rate']
    else:
        defaults['model_type'] = 'prevalence'
        # Add suggested prevalence if provided
        if disease_classification and disease_classification.get('suggested_prevalence'):
            defaults['known_values']['prevalence'] = disease_classification['suggested_prevalence']

    # Build STRICT structured prompts with citation requirements
    model_type_label = "INCIDENCE" if is_incidence_model else "PREVALENCE"
    
    # Define metric list based on model type
    metrics_list = defaults['metrics']
    metrics_template = '\n'.join([f'- {m}' for m in metrics_list])
    
    # Define known values section based on model type
    if is_incidence_model:
        known_values_template = """incidence: [decimal value - annual new cases per 100,000 population, e.g., 0.00025 = 25 per 100k]
diagnosed: [decimal value between 0 and 1]
treated: [decimal value between 0 and 1]"""
        citations_template = """incidence: [Source: Author, Year, Journal/Study name, DOI/URL if available]
diagnosed: [Source: Author, Year, Journal/Study name, DOI/URL if available]
treated: [Source: Author, Year, Journal/Study name, DOI/URL if available]"""
    else:
        known_values_template = """prevalence: [decimal value between 0 and 1]
diagnosed: [decimal value between 0 and 1]
treated: [decimal value between 0 and 1]"""
        citations_template = """prevalence: [Source: Author, Year, Journal/Study name, DOI/URL if available]
diagnosed: [Source: Author, Year, Journal/Study name, DOI/URL if available]
treated: [Source: Author, Year, Journal/Study name, DOI/URL if available]"""
    
    system_prompt = f"""You are an expert epidemiology forecasting assistant.

{'='*60}
DISEASE MODEL TYPE: {model_type_label}-BASED
{'='*60}

{'''FOR INCIDENCE-BASED DISEASES (CANCER/ONCOLOGY):
- Focus on ANNUAL NEW CASES (incident cases per year)
- Incidence Rate = New diagnoses per 100,000 population per year
- Example: Bladder cancer incidence ~20 per 100,000 = 0.0002 (as decimal)
- Diagnosed Rate = % of incident cases detected
- Treated Rate = % of diagnosed cases receiving treatment
''' if is_incidence_model else '''FOR PREVALENCE-BASED DISEASES (CHRONIC):
- Focus on TOTAL LIVING POPULATION with condition
- Prevalence Rate = % of total population currently with disease
- Example: 10% diabetes prevalence = 0.10 (as decimal)
- Diagnosed Rate = % of prevalent population diagnosed
- Treated Rate = % of diagnosed population on treatment
'''}

CRITICAL INSTRUCTIONS:
1. You MUST respond ONLY in the exact format specified below
2. Use EXACT metric names as shown in the template
3. All rates must be expressed as DECIMALS (not percentages)
4. Follow the exact section headers and format
5. PROVIDE CITATIONS for all epidemiological rates
6. ALWAYS include AGE_SEGMENTS section with age distribution data from research

Your response must follow this EXACT structure:

METRICS:
{metrics_template}

BASE_YEAR: [4-digit year]

GROWTH_RATES:
Population: [decimal value]

KNOWN_VALUES:
{known_values_template}

AGE_SEGMENTS:
[Age group name]: [decimal rate], [source citation]
[Age group name]: [decimal rate], [source citation]
[Age group name]: [decimal rate], [source citation]

CITATIONS:
{citations_template}

ASSUMPTIONS:
[Brief text explanation of assumptions]

IMPORTANT RULES:
- Metric names must be EXACTLY as shown above (case-sensitive)
- Rates are ALWAYS decimals: 0.36 means 36%, 0.85 means 85%
- Each metric on its own line starting with "-"
- Do not add extra metrics unless specifically requested
- Do not use percentages (%) in numeric values
- ALWAYS provide credible research citations for epidemiological rates
- Citations should include author/organization, year, study/journal name
- ALWAYS include AGE_SEGMENTS section with age distribution data from research
- Age segments should sum to approximately 1.0 (100%)
- Use format like "Age 18-44", "Age 45-64", "Age 65+" for age group names


IMPORTANT RULES:
- Metric names must be EXACTLY as shown above (case-sensitive)
- Rates are ALWAYS decimals: 0.36 means 36%, 0.85 means 85%
- Each metric on its own line starting with "-"
- Do not add extra metrics unless specifically requested
- Do not use percentages (%) in numeric values
- ALWAYS provide credible research citations for epidemiological rates
- Citations should include author/organization, year, study/journal name
"""

    user_prompt = f"""Based on the following input, provide {model_type_label}-BASED epidemiology forecast parameters WITH RESEARCH CITATIONS:

Disease: {input_data.get('disease_name', 'Not specified')}
Geography: {input_data.get('geography', 'Not specified')}
Population: {input_data.get('population', 'Not provided')}
Age Range: {input_data.get('age_range', 'Not specified')}
Known Metrics: {input_data.get('known_metrics', {})}
Additional Notes: {input_data.get('notes', 'None')}

{'''CRITICAL: This is a CANCER/INCIDENCE-BASED disease.
- Provide ANNUAL INCIDENCE RATE (new cases per 100,000 population per year)
- Express as decimal: 20 cases per 100k = 0.0002
- Use NCI SEER, cancer registries, or oncology literature for incidence rates
- DO NOT use prevalence for cancer - use incidence only
''' if is_incidence_model else '''CRITICAL: This is a CHRONIC/PREVALENCE-BASED disease.
- Provide TOTAL PREVALENCE RATE (% of population currently with condition)
- Express as decimal: 10% = 0.10
- Use CDC, WHO, or epidemiological studies for prevalence rates
'''}

Provide reasonable epidemiological rates for this disease and geography.
Use the EXACT format specified in your instructions.

If user provided specific rates ({'incidence' if is_incidence_model else 'prevalence'}, diagnosed, treated), use those values.
Otherwise, provide evidence-based estimates for this disease WITH CITATIONS FROM:
- Peer-reviewed medical journals
{'''- NCI SEER Cancer Statistics
- National/International cancer registries
- Oncology guidelines and textbooks''' if is_incidence_model else '''- CDC reports
- WHO reports
- National health databases'''}
- Recent epidemiological studies (prefer within last 5 years)

CRITICAL: ALWAYS search for and include AGE DISTRIBUTION data:
- Find published age distribution/breakdown for this disease in the specified geography
- Include 2-4 age groups that cover the full patient population
- Common age groups: "Age 18-44", "Age 45-64", "Age 65+" or disease-specific ranges
- Rates should sum to approximately 1.0 (100%)
- Include citation for age distribution source

Remember:
- Express all rates as DECIMALS (e.g., 0.14 for 14%, not 14 or 14%)
- Use EXACT metric names from the template
- Population growth should be annual rate (e.g., 0.007 for 0.7% annual growth)
- INCLUDE SPECIFIC CITATIONS for each rate you provide
- DO NOT skip the AGE_SEGMENTS section
"""

    try:
        if not all([AZURE_OPENAI_ENDPOINT, DEPLOYMENT_CHAT, OPENAI_API_KEY]):
            raise ValueError("Azure credentials not configured")

        response = client.chat.completions.create(
            model=DEPLOYMENT_CHAT,
            messages=[
                {'role': 'system', 'content': system_prompt},
                {'role': 'user', 'content': user_prompt}
            ],
            temperature=0  # Lower temperature for more consistent responses
        )

        raw = response.choices[0].message.content
        # Show raw response in UI for debugging
        st.session_state['ai_raw_response'] = raw

        # Parse response with strict section parsing
        sections = {}
        current_section = None
        
        for line in raw.splitlines():
            line_stripped = line.strip()
            if not line_stripped:
                continue
            
            # Check for section headers
            if line_stripped.endswith(':'):
                header = line_stripped[:-1].upper()
                if header in ['METRICS', 'BASE_YEAR', 'GROWTH_RATES', 'KNOWN_VALUES', 'ASSUMPTIONS', 'CITATIONS']:
                    current_section = header
                    sections[current_section] = []
                    continue
            
            # Check for section headers with content on same line
            for header in ['METRICS', 'BASE_YEAR', 'GROWTH_RATES', 'KNOWN_VALUES', 'ASSUMPTIONS', 'CITATIONS']:
                if line_stripped.upper().startswith(header + ':'):
                    current_section = header
                    sections[current_section] = []
                    # Extract content after the header
                    content = line_stripped[len(header)+1:].strip()
                    if content:
                        sections[current_section].append(content)
                    break
            else:
                # Add line to current section
                if current_section:
                    sections[current_section].append(line_stripped)

        # st.write("DEBUG - Parsed sections:", sections)

        result = defaults.copy()

        # METRICS - extract clean metric names
        if 'METRICS' in sections:
            parsed_metrics = []
            for item in sections['METRICS']:
                # Remove bullet points, numbers, extra whitespace
                cleaned = item.strip()
                for prefix in ['-', '•', '*', '1.', '2.', '3.', '4.', '5.', '6.', '7.']:
                    if cleaned.startswith(prefix):
                        cleaned = cleaned[len(prefix):].strip()
                        break
                
                # CRITICAL: Remove any values that might be appended after a colon
                # e.g., "Total Population: 335000000" -> "Total Population"
                if ':' in cleaned:
                    cleaned = cleaned.split(':')[0].strip()
                
                if cleaned and len(cleaned) > 2:  # Avoid empty or very short strings
                    parsed_metrics.append(cleaned)
            
            if parsed_metrics:
                result['metrics'] = parsed_metrics
                # st.write(f"DEBUG - Parsed metrics: {parsed_metrics}")

        # BASE_YEAR
        if 'BASE_YEAR' in sections and sections['BASE_YEAR']:
            try:
                year_text = sections['BASE_YEAR'][0]
                # Extract 4-digit year
                year_match = re.search(r'\b(20\d{2})\b', year_text)
                if year_match:
                    result['base_year'] = int(year_match.group(1))
                    # st.write(f"DEBUG - Parsed base_year: {result['base_year']}")
            except Exception as e:
                st.warning(f"Could not parse BASE_YEAR: {e}")

        # GROWTH_RATES
        if 'GROWTH_RATES' in sections:
            for line in sections['GROWTH_RATES']:
                if ':' in line:
                    k, v = [p.strip() for p in line.split(':', 1)]
                    try:
                        # Remove any % signs and convert
                        v_clean = v.replace('%', '').strip()
                        rate = float(v_clean)
                        # If the number is > 1, assume it was given as percentage
                        if rate > 1:
                            rate = rate / 100.0
                        result['growth_rates'][k] = rate
                        # st.write(f"DEBUG - Parsed growth rate: {k} = {rate}")
                    except Exception as e:
                        st.warning(f"Could not parse growth rate '{line}': {e}")

        # KNOWN_VALUES
        if 'KNOWN_VALUES' in sections:
            for line in sections['KNOWN_VALUES']:
                if ':' in line:
                    k, v = [p.strip() for p in line.split(':', 1)]
                    try:
                        # Remove any % signs and convert
                        v_clean = v.replace('%', '').strip()
                        val = float(v_clean)
                        # If the number is > 1, assume it was given as percentage
                        if val > 1:
                            val = val / 100.0
                        result['known_values'][k.lower()] = val
                        # st.write(f"DEBUG - Parsed known value: {k.lower()} = {val}")
                    except Exception as e:
                        st.warning(f"Could not parse known value '{line}': {e}")

        # CITATIONS - NEW
        if 'CITATIONS' in sections:
            for line in sections['CITATIONS']:
                if ':' in line:
                    # Split only on first colon to preserve colons in citations
                    parts = line.split(':', 1)
                    if len(parts) == 2:
                        k, citation = [p.strip() for p in parts]
                        result['citations'][k.lower()] = citation
                        # st.write(f"DEBUG - Parsed citation: {k.lower()} = {citation}")

        # ASSUMPTIONS
        if 'ASSUMPTIONS' in sections:
            result['assumptions'] = '\n'.join(sections['ASSUMPTIONS'])
            # st.write(f"DEBUG - Parsed assumptions: {result['assumptions'][:100]}...")

        # Ensure population is returned
        result['population'] = input_data.get('population', result['population'])
        
        # Final safety check - ensure population is never None or zero
        if result['population'] is None or result['population'] == 0:
            result['population'] = 50000000  # Default 50M population

        # Normalize any keys returned by AI into canonical keys
        result['known_values'] = _normalize_known_values(result.get('known_values', {}))
        
        # st.write("DEBUG - Final known_values:", result['known_values'])
        # st.write("DEBUG - Final citations:", result.get('citations', {}))
        # st.write("DEBUG - Final metrics:", result['metrics'])

        return result

    except Exception as e:
        # store last raw error
        st.warning(f"AI unavailable or parse error: {e}. Using defaults.")
        st.session_state['ai_raw_response'] = str(e)
        return defaults


def funnel_agent_v2(input_data: Dict, ai_assumptions: Dict) -> Dict:
    """Enhanced AI agent for suggesting disease-specific metrics and patient segments."""
    
    system_prompt = """You are an expert epidemiology and pharmaceutical analyst.

ANALYZE the disease and suggest relevant patient segments and metrics.

For the given disease, suggest relevant patient segments in this EXACT format:

[Source Metric] -> [New Segment]: [rate as decimal] | [rationale]

EXAMPLES:
- Diagnosed Population -> Severe Hypertension: 0.25 | Stage 2+ hypertension requiring intensive treatment
- Treated Population -> Poor Glycemic Control: 0.30 | HbA1c >8% despite treatment
- Diagnosed Population -> Reduced Ejection Fraction: 0.45 | HFrEF patients eligible for specific therapies

Return 5-7 specific suggestions for the given disease."""

    metrics = ai_assumptions.get('metrics', [])
    disease = input_data.get('disease_name', 'Unknown')
    
    user_prompt = f"""Disease: {disease}
Geography: {input_data.get('geography', 'Unknown')}
Current metrics pipeline: {metrics}

Suggest 5-7 clinically relevant patient segments/subgroups for {disease} forecasting.
Focus on segments that would have different:
- Treatment approaches
- Market access patterns  
- Clinical outcomes
- Pharmaceutical targeting

Use the exact format: [Source] -> [Segment]: [rate] | [rationale]"""

    try:
        response = client.chat.completions.create(
            model=DEPLOYMENT_CHAT,
            messages=[
                {'role': 'system', 'content': system_prompt},
                {'role': 'user', 'content': user_prompt}
            ],
            temperature=0.3,
            max_tokens=1200
        )
        
        result = response.choices[0].message.content.strip()
        
        # Parse the response line by line
        suggestions = []
        lines = result.split('\n')
        
        for line in lines:
            line = line.strip()
            if '->' in line and ':' in line and '|' in line:
                try:
                    # Parse: "Source -> Segment: rate | rationale"
                    parts = line.split('->')
                    if len(parts) >= 2:
                        source = parts[0].strip().lstrip('- ')
                        rest = parts[1]
                        
                        colon_split = rest.split(':', 1)
                        if len(colon_split) >= 2:
                            segment_name = colon_split[0].strip()
                            value_and_rationale = colon_split[1].strip()
                            
                            pipe_split = value_and_rationale.split('|', 1)
                            if len(pipe_split) >= 2:
                                rate_str = pipe_split[0].strip()
                                rationale = pipe_split[1].strip()
                                
                                # Parse rate
                                try:
                                    rate = float(rate_str.replace('%', ''))
                                    if rate > 1:
                                        rate = rate / 100.0
                                    
                                    suggestions.append({
                                        'source_population': source,
                                        'segment_name': segment_name,
                                        'rate': rate,
                                        'rationale': rationale,
                                        'segment_type': 'proportion'
                                    })
                                except ValueError:
                                    continue
                except Exception:
                    continue
        
        if not suggestions:
            # Fallback to disease-specific defaults
            suggestions = get_disease_specific_defaults(disease)
        
        return {
            'suggested_segments': suggestions,
            'confidence': 'high' if len(suggestions) >= 3 else 'medium'
        }

    except Exception as e:
        st.error(f"Error in funnel agent: {str(e)}")
        # Fallback to disease-specific defaults
        suggestions = get_disease_specific_defaults(disease)
        return {
            'suggested_segments': suggestions,
            'confidence': 'low'
        }

def get_disease_specific_defaults(disease: str) -> Dict:
    """Fallback disease-specific segment defaults."""
    disease_lower = disease.lower() if disease else ''
    
    # Disease-specific defaults
    if any(term in disease_lower for term in ['hypertension', 'blood pressure', 'htn']):
        return {
            'Diagnosed Population -> High Severity': 0.30,
            'Diagnosed Population -> Medium Severity': 0.45, 
            'Diagnosed Population -> Low Severity': 0.25,
            'Treated Population -> Controlled BP': 0.60,
            'Treated Population -> Uncontrolled BP': 0.40
        }
    elif any(term in disease_lower for term in ['diabetes', 'diabetic']):
        return {
            'Diagnosed Population -> Type 1 Diabetes': 0.10,
            'Diagnosed Population -> Type 2 Diabetes': 0.90,
            'Treated Population -> Good Glycemic Control': 0.55,
            'Treated Population -> Poor Glycemic Control': 0.45
        }
    elif any(term in disease_lower for term in ['heart failure', 'hf', 'cardiac']):
        return {
            'Diagnosed Population -> HFrEF': 0.50,
            'Diagnosed Population -> HFpEF': 0.50,
            'Diagnosed Population -> NYHA Class I-II': 0.70,
            'Diagnosed Population -> NYHA Class III-IV': 0.30
        }
    else:
        # Generic severity classification
        return {
            'Diagnosed Population -> High Severity': 0.25,
            'Diagnosed Population -> Medium Severity': 0.50,
            'Diagnosed Population -> Low Severity': 0.25,
            'Treated Population -> Treatment Responders': 0.70,
            'Treated Population -> Treatment Non-responders': 0.30
        }


def generate_forecast_v2(ai_assumptions: Dict, start_year: int, end_year: int, frequency: str = 'yearly') -> pd.DataFrame:
    """Generate epidemiology forecast based on AI assumptions and time range."""
    # Normalize frequency to lowercase for internal processing
    freq_normalized = frequency.lower()
    
    freq = 'YE' if freq_normalized == 'yearly' else 'QE' if freq_normalized == 'quarterly' else 'ME'
    periods = pd.date_range(start=f"{start_year}-01-01", end=f"{end_year}-12-31", freq=freq)
    period_labels = [p.strftime('%Y') if freq_normalized == 'yearly' else p.to_period('Q').strftime('%Y-Q%q') if freq_normalized == 'quarterly' else p.strftime('%Y-%m') for p in periods]

    def safe_float_convert(value, default=0.0):
        """Safely convert value to float"""
        if value is None:
            return default
        try:
            return float(value)
        except (ValueError, TypeError):
            return default
    
    metrics = ai_assumptions.get('metrics', [])
    st.write(f"Generating forecast for metrics: {metrics}")
    base_pop = safe_float_convert(ai_assumptions.get('population'), 1000000.0)
    growths = ai_assumptions.get('growth_rates', {})
    known = ai_assumptions.get('known_values', {})
    derived = ai_assumptions.get('derived_metrics', {})
    base_year = ai_assumptions.get('base_year', start_year)

    if not base_pop:
        base_pop = 1000000.0

    # Calculate time array based on frequency
    if freq_normalized == 'yearly':
        t = np.arange(len(period_labels))
    elif freq_normalized == 'quarterly':
        t = np.arange(len(period_labels)) / 4.0
    else:
        t = np.arange(len(period_labels)) / 12.0

    results = {}

    def get_growth_rate(metric_type: str, default: float = 0.0):
        """Get the growth rate for a metric type"""
        for key in growths:
            if metric_type.lower() in key.lower():
                return safe_float_convert(growths[key], default)
        return default

    def find_rate_metric(metric_type: str, idx_limit: int):
        """Find corresponding rate metric in earlier metrics"""
        for m in metrics[:idx_limit]:
            if metric_type.lower() in m.lower() and ('rate' in m.lower() or '%' in m):
                return m
        return None

    def get_base_rate_value(metric_type: str, default: float):
        """Get the base rate value from known_values"""
        st.write(f"    Looking for rate value for '{metric_type}' in known_values: {known}")
        
        def safe_float(val):
            """Safely convert to float, return default if None or invalid"""
            return safe_float_convert(val, default)
        
        # Try exact key match
        if metric_type in known:
            val = safe_float(known[metric_type])
            st.write(f"    Found exact match: {metric_type} = {val}")
            return val
        
        # Try with _rate suffix
        rate_key = f"{metric_type}_rate"
        if rate_key in known:
            val = safe_float(known[rate_key])
            st.write(f"    Found with _rate suffix: {rate_key} = {val}")
            return val
        
        # Try case-insensitive partial match
        for key, value in known.items():
            if metric_type.lower() in key.lower():
                val = safe_float(value)
                st.write(f"    Found partial match: {key} = {val}")
                return val
        
        st.write(f"    No match found, using default: {default}")
        return default

    def classify_metric(metric_name: str, idx: int):
        """Classify what type of metric this is based on its name"""
        name_lower = metric_name.lower()
        
        # Check if it's a rate metric
        is_rate = 'rate' in name_lower or 'percentage' in name_lower or '%' in name_lower
        
        # Classify the type
        if 'population' in name_lower and not any(kw in name_lower for kw in ['prevalent', 'prevalence', 'diagnosed', 'treated', 'treatment']):
            return ('total_population', is_rate)
        elif ('prevalent' in name_lower or 'prevalence' in name_lower):
            return ('prevalence', is_rate)
        elif 'diagnosed' in name_lower or 'diagnosis' in name_lower:
            return ('diagnosed', is_rate)
        elif 'treated' in name_lower or 'treatment' in name_lower or 'on treatment' in name_lower:
            return ('treated', is_rate)
        else:
            return ('unknown', is_rate)
    
    def find_source_population(source_name: str):
        """Find source population in results with flexible matching"""
        # Try exact match first
        if source_name in results:
            return results[source_name]
        
        # Try case-insensitive match
        source_lower = source_name.lower()
        for key in results.keys():
            if key.lower() == source_lower:
                return results[key]
        
        # Try partial match - look for key words
        source_keywords = source_lower.replace('population', '').replace('count', '').strip().split()
        for key in results.keys():
            key_lower = key.lower()
            # Check if all source keywords appear in the result key
            if all(keyword in key_lower for keyword in source_keywords if keyword):
                st.write(f"    Matched '{source_name}' to '{key}' via keywords")
                return results[key]
        
        # Special handling for common cases
        if 'treated' in source_lower:
            for key in results.keys():
                if 'treated' in key.lower() and 'population' in key.lower() and '%' not in key:
                    st.write(f"    Matched '{source_name}' to '{key}' (treated population)")
                    return results[key]
        
        if 'diagnosed' in source_lower:
            for key in results.keys():
                if 'diagnosed' in key.lower() and ('population' in key.lower() or 'cases' in key.lower() or 'count' in key.lower()) and '%' not in key:
                    st.write(f"    Matched '{source_name}' to '{key}' (diagnosed population/cases)")
                    return results[key]
        
        # Handle incident cases
        if 'incident' in source_lower or 'incidence' in source_lower:
            for key in results.keys():
                if ('incident' in key.lower() or 'incidence' in key.lower()) and 'cases' in key.lower() and '%' not in key:
                    st.write(f"    Matched '{source_name}' to '{key}' (incident cases)")
                    return results[key]
        
        return None

    # Process each metric in order
    for idx, metric in enumerate(metrics):
        metric_lower = metric.lower()
        
        st.write(f"\n{'='*60}")
        st.write(f"Processing [{idx}]: '{metric}'")
        
        # ===== DERIVED/CUSTOM METRICS (check FIRST!) =====
        if metric in derived:
            dm = derived[metric]
            source = dm.get('source')
            base_rate = safe_float_convert(dm.get('base_rate'), 0.0)
            growth_rate_val = safe_float_convert(dm.get('growth_rate'), 0.0)
            is_rate = dm.get('is_rate', False)
            
            st.write(f"  ✓ MATCHED: Custom/Derived metric")
            st.write(f"    Source: '{source}'")
            st.write(f"    Base rate: {base_rate}")
            st.write(f"    Growth rate: {growth_rate_val}")
            st.write(f"    Is rate metric: {is_rate}")
            
            # CRITICAL CHECK: Warn if base_rate is 0
            if base_rate == 0.0 and not is_rate:
                st.error(f"    ⚠️ ERROR: base_rate is 0.0 for count metric '{metric}'!")
                st.error(f"    This will produce all zeros. Check derived_metrics configuration.")
            
            if is_rate:
                # This is a RATE metric (e.g., "Eligible %") - should be constant or growing
                growth_type = dm.get('growth_type', 'proportional')
                
                if growth_type == 'linear':
                    # Linear growth rate: base_rate + (growth_rate * time)
                    results[metric] = base_rate + (growth_rate_val * t)
                    st.write(f"    Calculated as linear rate growth: {base_rate} + ({growth_rate_val} * t)")
                elif growth_rate_val == 0:
                    # Constant rate - just repeat the base_rate for all periods
                    results[metric] = np.full_like(t, base_rate, dtype=float)
                    st.write(f"    Calculated as constant rate: {base_rate}")
                else:
                    # Exponential rate growth
                    results[metric] = base_rate * (1 + growth_rate_val) ** t
                    st.write(f"    Calculated as exponential rate growth: {base_rate} * (1 + {growth_rate_val})^t")
                st.write(f"    Result[0] = {results[metric][0]:.4f}")
            else:
                # This is a POPULATION/COUNT metric (e.g., "Eligible Count")
                growth_type = dm.get('growth_type', 'proportional')
                
                if growth_type == 'absolute_linear':
                    # Absolute linear metric - grows independently
                    base_value = safe_float_convert(dm.get('base_value'), 1000)
                    results[metric] = base_value + (growth_rate_val * base_value * t)
                    st.write(f"    Calculated as absolute linear growth: {base_value} + ({growth_rate_val} * {base_value} * t)")
                    st.write(f"    Result[0] = {results[metric][0]:.0f}")
                    continue
                
                # Standard population metric - depends on source population and rate
                # FIRST: Check if this metric explicitly specifies its rate_metric
                explicit_rate_metric = dm.get('rate_metric')
                
                if explicit_rate_metric:
                    # Metric explicitly links to a rate metric - use that
                    st.write(f"    Explicit rate metric specified: '{explicit_rate_metric}'")
                    if explicit_rate_metric in results:
                        rate_values = results[explicit_rate_metric]
                        st.write(f"    ✓ Using rate from '{explicit_rate_metric}': {rate_values[0]:.4f}")
                    else:
                        # Rate metric not calculated yet - shouldn't happen with proper ordering
                        st.warning(f"    ⚠️ Rate metric '{explicit_rate_metric}' not calculated yet, using base_rate")
                        rate_values = np.full_like(t, base_rate, dtype=float)
                else:
                    # No explicit rate_metric - try to infer from name
                    base_metric_name = metric.replace(' Population', '').replace(' population', '').replace(' Count', '').strip()
                    potential_rate_name = f"{base_metric_name} %"
                    
                    st.write(f"    Looking for rate metric: '{potential_rate_name}'")
                    
                    # Check if there's a rate metric for this population
                    if potential_rate_name in results:
                        # Use the calculated rate metric
                        rate_values = results[potential_rate_name]
                        st.write(f"    ✓ Found rate metric '{potential_rate_name}', using its values")
                        st.write(f"      Rate[0] = {rate_values[0]:.4f}")
                    elif potential_rate_name in metrics[:idx]:
                        # Rate metric exists but hasn't been calculated yet (shouldn't happen with proper ordering)
                        st.warning(f"    ⚠️ Rate metric '{potential_rate_name}' exists but not calculated yet")
                        rate_values = base_rate * (1 + growth_rate_val) ** t
                    else:
                        # No rate metric, use base_rate directly
                        st.write(f"    No rate metric found, using base_rate: {base_rate}")
                        if base_rate == 0.0:
                            st.error(f"    ⚠️ base_rate is 0.0! This will produce zeros.")
                        rate_values = np.full_like(t, base_rate, dtype=float)
                
                # Get source population with flexible matching
                st.write(f"    Looking for source population: '{source}'")
                source_vals = find_source_population(source)
                
                if source_vals is None:
                    st.error(f"    ❌ ERROR: Source '{source}' not found in results!")
                    st.write(f"    Available populations: {[k for k in results.keys() if 'population' in k.lower() or 'count' in k.lower()]}")
                    source_vals = np.zeros_like(t)
                else:
                    st.write(f"    ✓ Found source population")
                    st.write(f"      Source[0] = {source_vals[0]:.0f}")
                
                # Calculate: source population × rate
                results[metric] = np.round(source_vals * rate_values, 0)
                st.write(f"    Formula: {source} × rate")
                if len(source_vals) > 0 and len(rate_values) > 0:
                    st.write(f"    ✓ Result[0] = {source_vals[0]:.0f} × {rate_values[0]:.4f} = {results[metric][0]:.0f}")
            continue
        
        # Classify the metric
        metric_type, is_rate = classify_metric(metric, idx)
        st.write(f"  Classified as: type='{metric_type}', is_rate={is_rate}")
        
        # ===== TOTAL POPULATION =====
        if metric_type == 'total_population' and not is_rate:
            st.write(f"  ✓ MATCHED: Total Population")
            pop_growth = get_growth_rate('population', 0.007)
            results[metric] = base_pop * (1 + pop_growth) ** t
            st.write(f"    base={base_pop}, growth={pop_growth}")
            st.write(f"    Result[0] = {results[metric][0]:.0f}")
            continue
        
        # ===== INCIDENCE RATE =====
        if ('incidence' in metric_lower or 'incident' in metric_lower) and is_rate:
            st.write(f"  ✓ MATCHED: Annual Incidence Rate")
            base_rate = get_base_rate_value('incidence', 0.0001)
            growth_rate = get_growth_rate('incidence_rate', 0.0)
            results[metric] = base_rate * (1 + growth_rate) ** t
            st.write(f"    base={base_rate}, growth={growth_rate}")
            st.write(f"    Result[0] = {results[metric][0]:.6f}")
            continue
        
        # ===== INCIDENT CASES =====
        if ('incidence' in metric_lower or 'incident' in metric_lower) and not is_rate and 'cases' in metric_lower:
            st.write(f"  ✓ MATCHED: Annual Incident Cases")
            
            # Find total population source
            total_pop = None
            for m in metrics[:idx]:
                m_type, m_is_rate = classify_metric(m, metrics.index(m))
                if m_type == 'total_population' and not m_is_rate and m in results:
                    total_pop = results[m]
                    st.write(f"    Found source population: '{m}'")
                    break
            
            if total_pop is None:
                st.write(f"    WARNING: No total population found, using base")
                total_pop = base_pop * (1 + get_growth_rate('population', 0.007)) ** t
            
            # Find incidence rate
            rate_metric = None
            for m in metrics[:idx]:
                if ('incidence' in m.lower() or 'incident' in m.lower()) and ('rate' in m.lower() or '%' in m):
                    rate_metric = m
                    break
            
            if rate_metric and rate_metric in results:
                rate = results[rate_metric]
                st.write(f"    Using rate metric: '{rate_metric}' = {rate[0]:.6f}")
            else:
                rate = get_base_rate_value('incidence', 0.0001) * np.ones_like(t)
                st.write(f"    Using rate value: {rate[0]:.6f}")
            
            results[metric] = np.round(total_pop * rate, 0)
            st.write(f"    Result[0] = {total_pop[0]:.0f} × {rate[0]:.6f} = {results[metric][0]:.0f}")
            continue
        
        # ===== PREVALENCE RATE =====
        if metric_type == 'prevalence' and is_rate:
            st.write(f"  ✓ MATCHED: Prevalence Rate")
            base_rate = get_base_rate_value('prevalence', 0.02)
            growth_rate = get_growth_rate('prevalence_rate', 0.0)
            results[metric] = base_rate * (1 + growth_rate) ** t
            st.write(f"    base={base_rate}, growth={growth_rate}")
            st.write(f"    Result[0] = {results[metric][0]:.4f}")
            continue
        
        # ===== PREVALENT POPULATION =====
        if metric_type == 'prevalence' and not is_rate:
            st.write(f"  ✓ MATCHED: Prevalent Population")
            
            # Find total population source
            total_pop = None
            for m in metrics[:idx]:
                m_type, m_is_rate = classify_metric(m, metrics.index(m))
                if m_type == 'total_population' and not m_is_rate and m in results:
                    total_pop = results[m]
                    st.write(f"    Found source population: '{m}'")
                    break
            
            if total_pop is None:
                st.write(f"    WARNING: No total population found, using base")
                total_pop = base_pop * (1 + get_growth_rate('population', 0.007)) ** t
            
            # Find prevalence rate
            rate_metric = find_rate_metric('prevalence', idx)
            if rate_metric and rate_metric in results:
                rate = results[rate_metric]
                st.write(f"    Using rate metric: '{rate_metric}' = {rate[0]:.4f}")
            else:
                rate = get_base_rate_value('prevalence', 0.02) * np.ones_like(t)
                st.write(f"    Using rate value: {rate[0]:.4f}")
            
            results[metric] = np.round(total_pop * rate, 0)
            st.write(f"    Result[0] = {total_pop[0]:.0f} × {rate[0]:.4f} = {results[metric][0]:.0f}")
            continue
        
        # ===== DIAGNOSED RATE =====
        if metric_type == 'diagnosed' and is_rate:
            st.write(f"  ✓ MATCHED: Diagnosed Rate")
            base_rate = get_base_rate_value('diagnosed', 0.85)
            growth_rate = get_growth_rate('diagnosed_rate', 0.0)
            results[metric] = base_rate * (1 + growth_rate) ** t
            st.write(f"    base={base_rate}, growth={growth_rate}")
            st.write(f"    Result[0] = {results[metric][0]:.4f}")
            continue
        
        # ===== DIAGNOSED POPULATION =====
        if metric_type == 'diagnosed' and not is_rate:
            st.write(f"  ✓ MATCHED: Diagnosed Population")
            
            # Try to find incident cases first (for incidence-based models)
            source_pop = None
            for m in reversed(metrics[:idx]):
                if ('incidence' in m.lower() or 'incident' in m.lower()) and 'cases' in m.lower() and m in results:
                    source_pop = results[m]
                    st.write(f"    Found source (incident cases): '{m}'")
                    break
            
            # If no incident cases, try prevalent population (for prevalence-based models)
            if source_pop is None:
                for m in reversed(metrics[:idx]):
                    m_type, m_is_rate = classify_metric(m, metrics.index(m))
                    if m_type == 'prevalence' and not m_is_rate and m in results:
                        source_pop = results[m]
                        st.write(f"    Found source (prevalent population): '{m}'")
                        break
            
            if source_pop is None:
                st.write(f"    WARNING: No source population found (neither incident cases nor prevalent population)")
                # Fallback to calculating from total population
                total_pop = None
                for m in reversed(metrics[:idx]):
                    m_type, m_is_rate = classify_metric(m, metrics.index(m))
                    if m_type == 'total_population' and not m_is_rate and m in results:
                        total_pop = results[m]
                        break
                
                if total_pop is not None:
                    prev_rate = get_base_rate_value('prevalence', 0.02)
                    source_pop = total_pop * prev_rate
                else:
                    source_pop = base_pop * get_base_rate_value('prevalence', 0.02) * np.ones_like(t)
            
            # Find diagnosed rate
            rate_metric = find_rate_metric('diagnosed', idx)
            if rate_metric and rate_metric in results:
                rate = results[rate_metric]
                st.write(f"    Using rate metric: '{rate_metric}' = {rate[0]:.4f}")
            else:
                rate = get_base_rate_value('diagnosed', 0.85) * np.ones_like(t)
                st.write(f"    Using rate value: {rate[0]:.4f}")
            
            results[metric] = np.round(source_pop * rate, 0)
            st.write(f"    Result[0] = {source_pop[0]:.0f} × {rate[0]:.4f} = {results[metric][0]:.0f}")
            continue
        
        # ===== TREATED RATE =====
        if metric_type == 'treated' and is_rate:
            st.write(f"  ✓ MATCHED: Treated Rate")
            base_rate = get_base_rate_value('treated', 0.78)
            growth_rate = get_growth_rate('treated_rate', 0.0)
            results[metric] = base_rate * (1 + growth_rate) ** t
            st.write(f"    base={base_rate}, growth={growth_rate}")
            st.write(f"    Result[0] = {results[metric][0]:.4f}")
            continue
        
        # ===== TREATED POPULATION =====
        if metric_type == 'treated' and not is_rate:
            st.write(f"  ✓ MATCHED: Treated Population")
            
            # Find diagnosed population source
            diagnosed_pop = None
            for m in reversed(metrics[:idx]):
                m_type, m_is_rate = classify_metric(m, metrics.index(m))
                if m_type == 'diagnosed' and not m_is_rate and m in results:
                    diagnosed_pop = results[m]
                    st.write(f"    Found source population: '{m}'")
                    break
            
            if diagnosed_pop is None:
                st.write(f"    WARNING: No diagnosed population found")
                diagnosed_pop = base_pop * get_base_rate_value('prevalence', 0.02) * get_base_rate_value('diagnosed', 0.85) * np.ones_like(t)
            
            # Find treated rate
            rate_metric = find_rate_metric('treated', idx)
            if rate_metric and rate_metric in results:
                rate = results[rate_metric]
                st.write(f"    Using rate metric: '{rate_metric}' = {rate[0]:.4f}")
            else:
                rate = get_base_rate_value('treated', 0.78) * np.ones_like(t)
                st.write(f"    Using rate value: {rate[0]:.4f}")
            
            results[metric] = np.round(diagnosed_pop * rate, 0)
            st.write(f"    Result[0] = {diagnosed_pop[0]:.0f} × {rate[0]:.4f} = {results[metric][0]:.0f}")
            continue
        
        # ===== FALLBACK: UNRECOGNIZED METRIC =====
        st.warning(f"  ⚠️ UNRECOGNIZED metric type: '{metric}'")
        st.write(f"    Using default: zeros")
        results[metric] = np.zeros_like(t)

    # Build DataFrame
    df_rows = {metric: results.get(metric, np.zeros_like(t)) for metric in metrics}
    df_final = pd.DataFrame(df_rows, index=period_labels).T

    # Round count-like rows to integers
    def _is_count_row(name: str) -> bool:
        nl = name.lower()
        return 'population' in nl or 'count' in nl

    count_rows = [r for r in df_final.index if _is_count_row(r)]
    if count_rows:
        for col in df_final.columns:
            for row in count_rows:
                df_final.loc[row, col] = int(round(df_final.loc[row, col], 0))

    return df_final

def normalize_derived_metrics(derived_metrics: Dict) -> Dict:
    """Ensure all derived metrics have the required structure"""
    normalized = {}
    
    for metric_name, config in derived_metrics.items():
        normalized_config = {
            'source': config.get('source', 'Total Population'),
            'base_rate': config.get('base_rate', config.get('base_value', 0.0)),
            'growth_rate': config.get('growth_rate', 0.0),
            'is_rate': config.get('is_rate', False),
            'growth_type': config.get('growth_type', 'proportional')
        }
        
        # For absolute metrics, also preserve base_value
        if 'base_value' in config:
            normalized_config['base_value'] = config['base_value']
            
        # For rate metrics with rate_metric reference
        if 'rate_metric' in config:
            normalized_config['rate_metric'] = config['rate_metric']
            
        normalized[metric_name] = normalized_config
    
    return normalized


def export_to_excel_v2(df_metrics: pd.DataFrame, ai_assumptions: Dict, filename: str) -> bytes:
    """Enhanced export with formulas for calculated fields"""
    writer = pd.ExcelWriter(filename, engine='xlsxwriter')
    workbook = writer.book
    
    # Summary sheet
    summary_data = {
        'Parameter': ['Disease', 'Geography', 'Base Year', 'Population', 'Prevalence Rate', 
                     'Diagnosed Rate', 'Treated Rate', 'Population Growth'],
        'Value': [
            ai_assumptions.get('disease_name', ''),
            ai_assumptions.get('geography', ''),
            ai_assumptions.get('base_year', ''),
            f"{ai_assumptions.get('population', 0):,}",
            f"{ai_assumptions.get('known_values', {}).get('prevalence', 0)*100:.2f}%",
            f"{ai_assumptions.get('known_values', {}).get('diagnosed', 0)*100:.2f}%",
            f"{ai_assumptions.get('known_values', {}).get('treated', 0)*100:.2f}%",
            f"{ai_assumptions.get('growth_rates', {}).get('Population', 0)*100:.3f}%"
        ]
    }
    summary_df = pd.DataFrame(summary_data)
    summary_df.to_excel(writer, sheet_name='Summary', index=False)
    
    # ===== FORECAST SHEET WITH FORMULAS =====
    worksheet = workbook.add_worksheet('Forecast Model')
    
    # Get data - USE ONLY ACTUAL METRICS FROM THE DATAFRAME INDEX
    metrics = [m for m in df_metrics.index if m.strip() and not m.startswith('===') and not m.startswith('  ')]  # Filter out separator rows
    base_pop = ai_assumptions.get('population', 0)
    base_year = ai_assumptions.get('base_year', datetime.now().year)
    growth_rates = ai_assumptions.get('growth_rates', {})
    known_values = ai_assumptions.get('known_values', {})
    derived_metrics = ai_assumptions.get('derived_metrics', {})
    
    # Column headers (years)
    years = list(df_metrics.columns)
    
    # Add formatting
    header_format = workbook.add_format({
        'bold': True,
        'bg_color': '#4472C4',
        'font_color': 'white',
        'border': 1,
        'align': 'center'
    })
    
    rate_format = workbook.add_format({
        'num_format': '0.00%',
        'border': 1
    })
    
    number_format = workbook.add_format({
        'num_format': '#,##0',
        'border': 1
    })
    
    constant_format = workbook.add_format({
        'num_format': '#,##0',
        'border': 1,
        'bg_color': '#E7E6E6'  # Gray background for constants
    })
    
    constant_rate_format = workbook.add_format({
        'num_format': '0.00%',
        'border': 1,
        'bg_color': '#E7E6E6'  # Gray background for constants
    })
    
    metric_name_format = workbook.add_format({
        'border': 1,
        'align': 'left',
        'valign': 'vcenter'
    })
    
    # ===== NEW: ADD PARAMETERS SECTION AT TOP =====
    worksheet.merge_range('A1:B1', 'MODEL PARAMETERS', header_format)
    worksheet.set_row(0, 20)
    
    # Base Population
    worksheet.write(1, 0, 'Base Population:', metric_name_format)
    worksheet.write(1, 1, base_pop, constant_format)
    base_pop_cell = 'B2'
    
    # Base Year
    worksheet.write(2, 0, 'Base Year:', metric_name_format)
    worksheet.write(2, 1, base_year, constant_format)
    
    # Population Growth Rate
    pop_growth_rate = growth_rates.get('Population', 0.007)
    worksheet.write(3, 0, 'Population Growth (annual):', metric_name_format)
    worksheet.write(3, 1, pop_growth_rate, constant_rate_format)
    pop_growth_cell = 'B4'
    
    # Base Rates
    worksheet.write(4, 0, 'Prevalence Rate:', metric_name_format)
    worksheet.write(4, 1, known_values.get('prevalence', 0.02), constant_rate_format)
    prev_rate_cell = 'B5'
    
    worksheet.write(5, 0, 'Diagnosed Rate:', metric_name_format)
    worksheet.write(5, 1, known_values.get('diagnosed', 0.85), constant_rate_format)
    diag_rate_cell = 'B6'
    
    worksheet.write(6, 0, 'Treated Rate:', metric_name_format)
    worksheet.write(6, 1, known_values.get('treated', 0.78), constant_rate_format)
    treat_rate_cell = 'B7'
    
    # Set column widths
    worksheet.set_column('A:A', 30)
    worksheet.set_column('B:B', 15)
    
    # ===== FORECAST TABLE STARTS AT ROW 9 (after parameters) =====
    forecast_start_row = 8
    
    # Write headers
    worksheet.write(forecast_start_row, 0, 'Metric', header_format)
    for col_idx, year in enumerate(years, start=1):
        worksheet.write(forecast_start_row, col_idx, year, header_format)
        worksheet.set_column(col_idx, col_idx, 15)
    
    # Helper function to determine if metric is a rate
    def is_rate_metric(metric_name):
        return 'rate' in metric_name.lower() or 'percentage' in metric_name.lower() or metric_name.endswith(' %')
    
    # Helper function to get Excel column letter
    def get_excel_column(col_num):
        """Convert column number to Excel letter (0=A, 1=B, etc.)"""
        result = ""
        while col_num >= 0:
            result = chr(col_num % 26 + 65) + result
            col_num = col_num // 26 - 1
        return result
    
    # Helper function to classify metric type
    def classify_metric_for_excel(metric_name):
        name_lower = metric_name.lower()
        
        is_rate = 'rate' in name_lower or metric_name.endswith(' %')
        
        if 'population' in name_lower and not any(kw in name_lower for kw in ['prevalent', 'prevalence', 'diagnosed', 'treated', 'treatment']):
            return 'total_population', is_rate
        elif 'prevalent' in name_lower or 'prevalence' in name_lower:
            return 'prevalence', is_rate
        elif 'diagnosed' in name_lower:
            return 'diagnosed', is_rate
        elif 'treated' in name_lower or 'treatment' in name_lower:
            return 'treated', is_rate
        else:
            return 'unknown', is_rate
    
    # Track row positions for formula references
    row_map = {}  # metric_name -> row_number
    
    # Write metrics and formulas
    for row_idx, metric in enumerate(metrics, start=forecast_start_row + 1):
        # Write metric name
        worksheet.write(row_idx, 0, metric, metric_name_format)
        row_map[metric] = row_idx + 1  # Excel row (1-indexed)
        
        # ===== CHECK FOR DERIVED/CUSTOM METRICS FIRST! =====
        # ===== CUSTOM/DERIVED METRICS (check FIRST!) =====
        is_derived = metric in derived_metrics
        
        if is_derived:
            # This is a custom/derived metric - handle it specially
            config = derived_metrics[metric]
            source_metric = config.get('source')
            base_rate = config.get('base_rate', 0.0)
            is_rate_metric = config.get('is_rate', False)
            
            if is_rate_metric:
                # Custom rate - constant value (gray background)
                for col_idx in range(1, len(years) + 1):
                    worksheet.write_number(row_idx, col_idx, base_rate, constant_rate_format)
            else:
                # Custom population - formula based on STORED relationship
                for col_idx in range(1, len(years) + 1):
                    col_letter = get_excel_column(col_idx)
                    
                    # Find source row
                    source_row = row_map.get(source_metric)
                    
                    # FIX: Check if there's a linked rate_metric in the config
                    rate_metric_name = config.get('rate_metric')
                    
                    if not rate_metric_name:
                        # Legacy: Try to construct rate metric name
                        # Remove " Count" or " Population" suffix and add " %" or " Rate"
                        if metric.endswith(' Count'):
                            base_name = metric[:-len(' Count')]
                            rate_metric_name = f"{base_name} %"
                        elif metric.endswith(' Population'):
                            base_name = metric[:-len(' Population')]
                            rate_metric_name = f"{base_name} Rate"
                        elif metric.endswith(' population'):
                            base_name = metric[:-len(' population')]
                            rate_metric_name = f"{base_name} Rate"
                        else:
                            # Fallback: just look for any metric with "Rate" and similar base
                            rate_metric_name = None
                    
                    # Find the rate row
                    rate_row = row_map.get(rate_metric_name) if rate_metric_name else None
                    
                    if source_row is not None and rate_row is not None:
                        # Formula: Source × Rate CELL (not constant)
                        formula = f'={col_letter}{source_row}*{col_letter}{rate_row}'
                        worksheet.write_formula(row_idx, col_idx, formula, number_format)
                    elif source_row is not None and base_rate > 0:
                        # No rate metric found - use constant base_rate
                        formula = f'={col_letter}{source_row}*{base_rate}'
                        worksheet.write_formula(row_idx, col_idx, formula, number_format)
                    elif source_row is not None:
                        # No rate at all - just reference source (shouldn't happen)
                        formula = f'={col_letter}{source_row}'
                        worksheet.write_formula(row_idx, col_idx, formula, number_format)
                    else:
                        # Source not found - ERROR! Write the actual value as fallback
                        try:
                            value = df_metrics.loc[metric, years[col_idx-1]]
                            worksheet.write_number(row_idx, col_idx, float(value), number_format)
                        except:
                            worksheet.write_number(row_idx, col_idx, 0, number_format)
            
            # Skip to next metric - don't process as standard metric
            continue
        
        # ===== STANDARD METRICS (only if not derived) =====
        metric_type, is_rate = classify_metric_for_excel(metric)
        
        # ===== TOTAL POPULATION =====
        if metric_type == 'total_population' and not is_rate:
            pop_growth = growth_rates.get('Population', 0.007)
            
            for col_idx, year in enumerate(years, start=1):
                year_val = int(year) if str(year).isdigit() else base_year
                years_from_base = year_val - base_year
                
                if col_idx == 1:
                    # First year - reference base population parameter
                    formula = f'={base_pop_cell}'
                    worksheet.write_formula(row_idx, col_idx, formula, constant_format)
                else:
                    # Formula: base_pop * (1 + growth)^years
                    col_letter = get_excel_column(1)
                    formula = f'={col_letter}{row_idx + 1}*POWER(1+{pop_growth_cell},{years_from_base})'
                    worksheet.write_formula(row_idx, col_idx, formula, number_format)
        
        # ===== PREVALENCE RATE =====
        elif metric_type == 'prevalence' and is_rate:
            for col_idx in range(1, len(years) + 1):
                # Reference the parameter cell for all years
                formula = f'={prev_rate_cell}'
                worksheet.write_formula(row_idx, col_idx, formula, constant_rate_format)
        
        # ===== PREVALENT POPULATION =====
        elif metric_type == 'prevalence' and not is_rate:
            # Formula: Total Population × Prevalence Rate
            for col_idx in range(1, len(years) + 1):
                col_letter = get_excel_column(col_idx)
                
                # Find Total Population row
                total_pop_row = None
                for m, r in row_map.items():
                    m_type, m_is_rate = classify_metric_for_excel(m)
                    if m_type == 'total_population' and not m_is_rate:
                        total_pop_row = r
                        break
                
                # Find Prevalence Rate row
                prev_rate_row = None
                for m, r in row_map.items():
                    m_type, m_is_rate = classify_metric_for_excel(m)
                    if m_type == 'prevalence' and m_is_rate:
                        prev_rate_row = r
                        break
                
                if total_pop_row and prev_rate_row:
                    formula = f'={col_letter}{total_pop_row}*{col_letter}{prev_rate_row}'
                    worksheet.write_formula(row_idx, col_idx, formula, number_format)
                else:
                    worksheet.write_number(row_idx, col_idx, 0, number_format)
        
        # ===== DIAGNOSED RATE =====
        elif metric_type == 'diagnosed' and is_rate:
            for col_idx in range(1, len(years) + 1):
                # Reference the parameter cell for all years
                formula = f'={diag_rate_cell}'
                worksheet.write_formula(row_idx, col_idx, formula, constant_rate_format)
        
        # ===== DIAGNOSED POPULATION =====
        elif metric_type == 'diagnosed' and not is_rate:
            # Formula: Prevalent Population × Diagnosed Rate
            for col_idx in range(1, len(years) + 1):
                col_letter = get_excel_column(col_idx)
                
                # Find Prevalent Population row
                prev_pop_row = None
                for m, r in row_map.items():
                    m_type, m_is_rate = classify_metric_for_excel(m)
                    if m_type == 'prevalence' and not m_is_rate:
                        prev_pop_row = r
                        break
                
                # Find Diagnosed Rate row
                diag_rate_row = None
                for m, r in row_map.items():
                    m_type, m_is_rate = classify_metric_for_excel(m)
                    if m_type == 'diagnosed' and m_is_rate:
                        diag_rate_row = r
                        break
                
                if prev_pop_row and diag_rate_row:
                    formula = f'={col_letter}{prev_pop_row}*{col_letter}{diag_rate_row}'
                    worksheet.write_formula(row_idx, col_idx, formula, number_format)
                else:
                    worksheet.write_number(row_idx, col_idx, 0, number_format)
        
        # ===== TREATED RATE =====
        elif metric_type == 'treated' and is_rate:
            for col_idx in range(1, len(years) + 1):
                # Reference the parameter cell for all years
                formula = f'={treat_rate_cell}'
                worksheet.write_formula(row_idx, col_idx, formula, constant_rate_format)
        
        # ===== TREATED POPULATION =====
        elif metric_type == 'treated' and not is_rate:
            # Formula: Diagnosed Population × Treated Rate
            for col_idx in range(1, len(years) + 1):
                col_letter = get_excel_column(col_idx)
                
                # Find Diagnosed Population row
                diag_pop_row = None
                for m, r in row_map.items():
                    m_type, m_is_rate = classify_metric_for_excel(m)
                    if m_type == 'diagnosed' and not m_is_rate:
                        diag_pop_row = r
                        break
                
                # Find Treated Rate row
                treat_rate_row = None
                for m, r in row_map.items():
                    m_type, m_is_rate = classify_metric_for_excel(m)
                    if m_type == 'treated' and m_is_rate:
                        treat_rate_row = r
                        break
                
                if diag_pop_row and treat_rate_row:
                    formula = f'={col_letter}{diag_pop_row}*{col_letter}{treat_rate_row}'
                    worksheet.write_formula(row_idx, col_idx, formula, number_format)
                else:
                    worksheet.write_number(row_idx, col_idx, 0, number_format)
        
        # ===== FALLBACK =====
        else:
            # Unknown metric - write static values from dataframe
            for col_idx, year in enumerate(years, start=1):
                try:
                    value = df_metrics.loc[metric, year]
                    if is_rate:
                        worksheet.write_number(row_idx, col_idx, float(value), rate_format)
                    else:
                        worksheet.write_number(row_idx, col_idx, float(value), number_format)
                except:
                    worksheet.write_number(row_idx, col_idx, 0, number_format)
    
    # Add instructions sheet
    instructions_worksheet = workbook.add_worksheet('Instructions')
    instructions = [
        ['Epidemiology Forecast Model - Instructions'],
        [''],
        ['UNDERSTANDING THE MODEL:'],
        [''],
        ['Gray Cells (Constants/Parameters):'],
        ['- At the top of the sheet are the key parameters (Base Population, Growth Rate, etc.)'],
        ['- These gray cells can be edited to test different scenarios'],
        ['- All rate metrics (Prevalence Rate, Diagnosed Rate, Treated Rate) reference these parameters'],
        ['- Changes to these parameters automatically cascade through the entire model'],
        [''],
        ['White Cells (Formulas):'],
        ['- Population metrics are calculated using formulas'],
        ['- Based on the cascade: Population → Prevalence → Diagnosed → Treated'],
        ['- Do not edit these cells directly - modify the parameters instead'],
        [''],
        ['HOW TO USE:'],
        [''],
        ['1. Modify gray parameter cells at the top to test scenarios'],
        ['   Example: Change Prevalence Rate from 0.36 to 0.40 (36% to 40%)'],
        [''],
        ['2. All population metrics will recalculate automatically'],
        ['   Example: Prevalent Population = Total Population × Prevalence Rate'],
        [''],
        ['3. The cascade flows through the entire model:'],
        ['   Total Pop → Prevalent Pop → Diagnosed Pop → Treated Pop → Custom Metrics'],
        [''],
        ['FORMULA STRUCTURE:'],
        [''],
        ['Total Population:'],
        ['   - Year 1: References Base Population parameter'],
        ['   - Other Years: Year 1 × (1 + Growth Rate)^years'],
        [''],
        ['Prevalence Rate:'],
        ['   - All years reference the Prevalence Rate parameter'],
        [''],
        ['Prevalent Population:'],
        ['   - Formula: Total Population × Prevalence Rate'],
        [''],
        ['Diagnosed Rate:'],
        ['   - All years reference the Diagnosed Rate parameter'],
        [''],
        ['Diagnosed Population:'],
        ['   - Formula: Prevalent Population × Diagnosed Rate'],
        [''],
        ['Treated Rate:'],
        ['   - All years reference the Treated Rate parameter'],
        [''],
        ['Treated Population:'],
        ['   - Formula: Diagnosed Population × Treated Rate'],
        [''],
        ['Custom Metrics:'],
        ['   - Custom Percentage (%): Constant value (editable)'],
        ['   - Custom Count: Source Population × Custom Percentage'],
        ['   - Note: Custom metrics use % and Count suffixes to distinguish from standard rates'],
        [''],
        ['SCENARIO TESTING:'],
        [''],
        ['Try these by modifying the parameter cells at the top:'],
        [''],
        ['Higher Prevalence Scenario:'],
        ['- Change Prevalence Rate to 0.40 (40%)'],
        ['- See how all downstream populations increase'],
        [''],
        ['Better Diagnosis Scenario:'],
        ['- Change Diagnosed Rate to 0.90 (90%)'],
        ['- Model improved screening programs'],
        [''],
        ['Improved Treatment Access:'],
        ['- Change Treated Rate to 0.85 (85%)'],
        ['- See impact of better healthcare access'],
        [''],
        ['Population Growth Scenarios:'],
        ['- Adjust Population Growth rate'],
        ['- Model demographic changes'],
        [''],
        ['See the "Research Citations" sheet for data sources.']
    ]
    
    for row_idx, row_data in enumerate(instructions):
        for col_idx, cell_value in enumerate(row_data):
            instructions_worksheet.write(row_idx, col_idx, cell_value)
    
    instructions_worksheet.set_column('A:A', 80)
    
    # Citations sheet (existing code)
    citations = ai_assumptions.get('citations', {})
    if citations:
        citation_data = []
        for metric, citation in citations.items():
            rate_value = known_values.get(metric)
            citation_data.append({
                'Metric': metric.replace('_', ' ').title(),
                'Value': f"{rate_value*100:.2f}%" if rate_value else 'N/A',
                'Citation': citation
            })
        
        derived = ai_assumptions.get('derived_metrics', {})
        for metric_name, config in derived.items():
            metric_base = metric_name.replace(' %', '').replace(' Count', '').replace(' Rate', '').replace(' Population', '').lower()
            if metric_base in citations and not any(c['Metric'].lower() == metric_base for c in citation_data):
                citation_data.append({
                    'Metric': metric_base.replace('_', ' ').title(),
                    'Value': f"{config['base_rate']*100:.2f}%",
                    'Citation': citations[metric_base]
                })
        
        if citation_data:
            citations_df = pd.DataFrame(citation_data)
            citations_df.to_excel(writer, sheet_name='Research Citations', index=False)
            
            worksheet_citations = writer.sheets['Research Citations']
            worksheet_citations.set_column('A:A', 20)
            worksheet_citations.set_column('B:B', 12)
            worksheet_citations.set_column('C:C', 80)
    
    # Custom metrics detail
    derived = ai_assumptions.get('derived_metrics', {})
    if derived:
        custom_data = []
        for name, config in derived.items():
            custom_data.append({
                'Metric': name,
                'Type': 'Percentage' if config['is_rate'] else 'Count',
                'Source': config['source'],
                'Base Value': f"{config['base_rate']*100:.2f}%",
                'Growth Rate': f"{config['growth_rate']*100:.2f}%"
            })
        custom_df = pd.DataFrame(custom_data)
        custom_df.to_excel(writer, sheet_name='Custom Metrics', index=False)
    
    # Assumptions text
    assumptions_df = pd.DataFrame({
        'Assumptions': [ai_assumptions.get('assumptions', 'N/A')]
    })
    assumptions_df.to_excel(writer, sheet_name='AI Assumptions', index=False)
    
    writer.close()
    
    with open(filename, 'rb') as f:
        return f.read()

def generate_brand_forecast(epi_forecast: pd.DataFrame, brand_config: Dict, 
                           start_year: int, end_year: int, events: list = None) -> pd.DataFrame:
    """Generate brand-specific forecast with uptake, adherence, revenue, and events (e.g., LoE)
    
    Args:
        epi_forecast: Epidemiology forecast dataframe
        brand_config: Brand configuration dictionary
        start_year: Forecast start year
        end_year: Forecast end year
        events: List of event dictionaries with structure:
            {
                'event_year': int,
                'event_type': str ('loss_of_exclusivity', 'competitor_entry', etc.),
                'post_event_peak_share': float,
                'years_to_peak': int,
                'curve_type': str (from UPTAKE_CURVES or EROSION_CURVES),
                'curve_category': str ('Erosion (Decline)' or 'Recovery (Growth)'),
                'description': str
            }
    """
    
    # Extract brand configuration
    brand_name = brand_config.get('brand_name', 'Brand X')
    launch_year = brand_config.get('launch_year', start_year)
    peak_share = brand_config.get('peak_share', 0.35)  # 35% as in your example
    uptake_type = brand_config.get('uptake_type', '7 Year Medium')
    adherence_rate = brand_config.get('adherence', 0.68)  # 68% as in your example
    annual_dose = brand_config.get('annual_dose', 360)  # 360 doses per year
    price_per_dose = brand_config.get('price_per_dose', 16)  # $16
    target_population = brand_config.get('target_population', 'Treated patients')
    
    # Get the uptake curve
    uptake_curve = UPTAKE_CURVES.get(uptake_type, UPTAKE_CURVES['7 Year Medium'])
    
    # Get years from epidemiology forecast
    years = [int(col) if str(col).isdigit() else col for col in epi_forecast.columns]
    
    # Initialize results dictionary
    results = {
        'Brand X Patient Share': [],
        'Patients on Brand X': [],
        'Adherence': [],
        'Adherent Patients': [],
        'Annual dose per patient': [],
        'Price per dose ($)': [],
        'Total adherence adjusted Volume': [],
        'Revenue in billions': []
    }
    
    # Find target population row in epi_forecast
    # ENHANCED: Better matching for custom metrics (e.g., "Eligible Count", "High Risk Count")
    target_pop_row = None
    
    # First try exact match
    if target_population in epi_forecast.index:
        target_pop_row = target_population
    else:
        # Clean both target and row names for flexible matching
        target_population_clean = target_population.lower().replace('population', '').replace('patients', '').replace('count', '').replace('%', '').strip()
        
        for row_name in epi_forecast.index:
            row_name_clean = row_name.lower().replace('population', '').replace('count', '').replace('patients', '').replace('%', '').strip()
            
            # Check for exact match or partial match
            if (target_population_clean == row_name_clean or
                target_population_clean in row_name_clean or 
                row_name_clean in target_population_clean or
                row_name.lower() == target_population.lower()):
                # Prefer Count metrics over % metrics for population targeting
                if 'count' in row_name.lower() or 'population' in row_name.lower():
                    target_pop_row = row_name
                    break
                elif not target_pop_row:  # Use % metric as fallback
                    target_pop_row = row_name
        
        # If still not found, try to find "Count" version of the target
        if not target_pop_row and not target_population.endswith('Count'):
            count_version = f"{target_population} Count"
            if count_version in epi_forecast.index:
                target_pop_row = count_version
    
    if not target_pop_row:
        # Fallback to Treated Population
        for row_name in epi_forecast.index:
            if 'treated' in row_name.lower() and ('population' in row_name.lower() or 'count' in row_name.lower()):
                target_pop_row = row_name
                st.warning(f"Could not find '{target_population}', using '{target_pop_row}' instead")
                break
    
    if not target_pop_row:
        st.error(f"ERROR: Could not find target population '{target_population}' in epidemiology forecast")
        st.write(f"Available populations: {list(epi_forecast.index)}")
        # Use first row as absolute fallback
        target_pop_row = epi_forecast.index[0]
    
    target_pop_data = epi_forecast.loc[target_pop_row]
    st.write(f"Brand targeting: {target_pop_row}")
    
    # Sort events by year if provided
    if events:
        events = sorted(events, key=lambda x: x['event_year'])
    
    # Calculate for each year
    for idx, year in enumerate(years):
        try:
            year_int = int(year) if str(year).isdigit() else start_year + idx
        except:
            year_int = start_year + idx
        
        # Check if any events apply to this year
        active_event = None
        if events:
            # Find the most recent event that has occurred
            for event in reversed(events):  # Check from most recent to oldest
                if year_int >= event['event_year']:
                    active_event = event
                    break  # Use the most recent applicable event
        
        if active_event:
            # POST-EVENT PERIOD (e.g., after LoE, competitor entry, etc.)
            years_since_event = year_int - active_event['event_year']
            event_curve_type = active_event.get('curve_type', '5 Year Medium Erosion')
            curve_category = active_event.get('curve_category', 'Erosion (Decline)')
            
            # Determine if it's uptake (new growth) or erosion (decline)
            if curve_category == 'Erosion (Decline)' or 'Erosion' in event_curve_type:
                # Erosion curve - declining share
                curve_data = EROSION_CURVES.get(event_curve_type, EROSION_CURVES.get('5 Year Medium Erosion', [1.0, 0.8, 0.6, 0.4, 0.2, 0.1]))
            else:
                # Uptake curve - recovering/growing share
                curve_data = UPTAKE_CURVES.get(event_curve_type, UPTAKE_CURVES.get('5 Year Medium', [0.17, 0.43, 0.66, 0.85, 1.00]))
            
            # Get multiplier from curve
            if years_since_event >= len(curve_data):
                uptake_multiplier = curve_data[-1]  # Final value
            else:
                uptake_multiplier = curve_data[years_since_event]
            
            # Calculate share based on post-event peak
            post_event_peak = active_event.get('post_event_peak_share', peak_share * 0.5)  # Default to 50% of original
            
            if curve_category == 'Erosion (Decline)':
                # For erosion: start from current pre-event share and decline toward post_event_peak
                # Get the share just before the event
                if years_since_event == 0:
                    # Year of event - use pre-event calculation
                    years_since_launch = year_int - launch_year
                    if years_since_launch < 0:
                        pre_event_share = 0.0
                    elif years_since_launch >= len(uptake_curve):
                        pre_event_share = peak_share
                    else:
                        pre_event_share = peak_share * uptake_curve[years_since_launch]
                    brand_share = pre_event_share  # Start of erosion
                else:
                    # Erode from peak_share toward post_event_peak
                    # Erosion curves go from 1.0 down to near 0
                    # We interpret: share = post_event_peak + (peak_share - post_event_peak) * erosion_multiplier
                    brand_share = post_event_peak + (peak_share - post_event_peak) * uptake_multiplier
            else:
                # For recovery: grow toward post_event_peak
                brand_share = post_event_peak * uptake_multiplier
            
        else:
            # PRE-EVENT PERIOD (original logic - normal uptake)
            years_since_launch = year_int - launch_year
            
            if years_since_launch < 0:
                # Before launch
                uptake_multiplier = 0.0
            elif years_since_launch >= len(uptake_curve):
                # After curve ends, assume 100%
                uptake_multiplier = 1.0
            else:
                # During uptake period
                uptake_multiplier = uptake_curve[years_since_launch]
            
            # Brand X Patient Share = Peak Share × Uptake Multiplier
            brand_share = peak_share * uptake_multiplier
        
        # Patients on Brand X = Treated Population × Brand Share
        def safe_float_convert_local(val, default=0.0):
            if val is None or pd.isna(val):
                return default
            try:
                return float(val)
            except (ValueError, TypeError):
                return default
        
        treated_patients = safe_float_convert_local(target_pop_data.iloc[idx])
        patients_on_brand = treated_patients * brand_share
        
        # Adherent Patients = Patients on Brand × Adherence Rate
        adherent_patients = patients_on_brand * adherence_rate
        
        # Total Adherence Adjusted Volume = Adherent Patients × Annual Dose
        total_volume = adherent_patients * annual_dose
        
        # Revenue in billions = (Total Volume × Price per Dose) / 1,000,000,000
        revenue_billions = (total_volume * price_per_dose) / 1_000_000_000
        
        # Store results
        results['Brand X Patient Share'].append(brand_share)
        results['Patients on Brand X'].append(patients_on_brand)
        results['Adherence'].append(adherence_rate)
        results['Adherent Patients'].append(adherent_patients)
        results['Annual dose per patient'].append(annual_dose)
        results['Price per dose ($)'].append(price_per_dose)
        results['Total adherence adjusted Volume'].append(total_volume)
        results['Revenue in billions'].append(revenue_billions)
    
    # Convert to DataFrame with years as columns
    df_results = pd.DataFrame(results, index=years).T
    
    return df_results

def export_brand_to_excel(epi_forecast: pd.DataFrame, brand_forecast: pd.DataFrame, 
                          brand_config: Dict, ai_assumptions: Dict, filename: str) -> bytes:
    """Export comprehensive brand forecast to Excel with formulas and epidemiology data"""
    
    writer = pd.ExcelWriter(filename, engine='xlsxwriter')
    workbook = writer.book
    
    # ===== DEFINE FORMATS =====
    title_format = workbook.add_format({
        'bold': True,
        'font_size': 14,
        'bg_color': '#2E75B6',
        'font_color': 'white',
        'align': 'center',
        'valign': 'vcenter',
        'border': 1
    })
    
    header_format = workbook.add_format({
        'bold': True,
        'bg_color': '#4472C4',
        'font_color': 'white',
        'border': 1,
        'align': 'center',
        'valign': 'vcenter',
        'text_wrap': True
    })
    
    brand_header_format = workbook.add_format({
        'bold': True,
        'bg_color': '#70AD47',
        'font_color': 'white',
        'border': 1,
        'align': 'center',
        'valign': 'vcenter',
        'text_wrap': True
    })
    
    config_label_format = workbook.add_format({
        'bold': True,
        'border': 1,
        'bg_color': '#D9E1F2',
        'align': 'right'
    })
    
    constant_format = workbook.add_format({
        'num_format': '#,##0',
        'border': 1,
        'bg_color': '#E7E6E6'
    })
    
    constant_rate_format = workbook.add_format({
        'num_format': '0.00%',
        'border': 1,
        'bg_color': '#E7E6E6'
    })
    
    constant_currency_format = workbook.add_format({
        'num_format': '$#,##0.00',
        'border': 1,
        'bg_color': '#E7E6E6'
    })
    
    number_format = workbook.add_format({
        'num_format': '#,##0',
        'border': 1
    })
    
    rate_format = workbook.add_format({
        'num_format': '0.00%',
        'border': 1
    })
    
    currency_format = workbook.add_format({
        'num_format': '$#,##0.00',
        'border': 1
    })
    
    revenue_format = workbook.add_format({
        'num_format': '$#,##0.0',
        'border': 1,
        'bold': True
    })
    
    metric_name_format = workbook.add_format({
        'border': 1,
        'align': 'left',
        'valign': 'vcenter'
    })
    
    section_header_format = workbook.add_format({
        'bold': True,
        'font_size': 12,
        'bg_color': '#44546A',
        'font_color': 'white',
        'border': 1,
        'align': 'left'
    })
    
    # Helper functions
    def get_excel_column(col_num):
        result = ""
        while col_num >= 0:
            result = chr(col_num % 26 + 65) + result
            col_num = col_num // 26 - 1
        return result
    
    def is_rate_metric(metric_name):
        name_lower = metric_name.lower()
        return 'rate' in name_lower or 'percentage' in name_lower or 'share' in name_lower or metric_name.endswith(' %')
    
    def classify_metric_for_excel(metric_name):
        name_lower = metric_name.lower()
        
        is_rate = 'rate' in name_lower or metric_name.endswith(' %')
        
        if 'population' in name_lower and not any(kw in name_lower for kw in ['prevalent', 'prevalence', 'diagnosed', 'treated', 'treatment']):
            return 'total_population', is_rate
        elif 'prevalent' in name_lower or 'prevalence' in name_lower:
            return 'prevalence', is_rate
        elif 'diagnosed' in name_lower:
            return 'diagnosed', is_rate
        elif 'treated' in name_lower or 'treatment' in name_lower:
            return 'treated', is_rate
        else:
            return 'unknown', is_rate
    
    # ===== SHEET 1: SUMMARY =====
    summary_sheet = workbook.add_worksheet('Summary')
    summary_sheet.set_column('A:A', 25)
    summary_sheet.set_column('B:B', 30)
    
    row = 0
    summary_sheet.merge_range(f'A{row+1}:B{row+1}', 'BRAND FORECAST SUMMARY', title_format)
    row += 2
    
    summary_sheet.write(row, 0, 'Disease:', config_label_format)
    summary_sheet.write(row, 1, ai_assumptions.get('disease_name', 'N/A'))
    row += 1
    
    summary_sheet.write(row, 0, 'Geography:', config_label_format)
    summary_sheet.write(row, 1, ai_assumptions.get('geography', 'N/A'))
    row += 1
    
    summary_sheet.write(row, 0, 'Base Year:', config_label_format)
    summary_sheet.write(row, 1, ai_assumptions.get('base_year', 'N/A'))
    row += 1
    
    summary_sheet.write(row, 0, 'Total Population:', config_label_format)
    summary_sheet.write(row, 1, ai_assumptions.get('population', 0), number_format)
    row += 2
    
    summary_sheet.merge_range(f'A{row+1}:B{row+1}', 'BRAND CONFIGURATION', section_header_format)
    row += 1
    
    summary_sheet.write(row, 0, 'Brand Name:', config_label_format)
    summary_sheet.write(row, 1, brand_config.get('brand_name', 'Brand X'))
    row += 1
    
    summary_sheet.write(row, 0, 'Launch Year:', config_label_format)
    summary_sheet.write(row, 1, brand_config.get('launch_year', 2025))
    row += 1
    
    summary_sheet.write(row, 0, 'Peak Market Share:', config_label_format)
    summary_sheet.write(row, 1, brand_config.get('peak_share', 0.35), rate_format)
    row += 1
    
    summary_sheet.write(row, 0, 'Uptake Curve:', config_label_format)
    summary_sheet.write(row, 1, brand_config.get('uptake_type', '7 Year Medium'))
    row += 1
    
    summary_sheet.write(row, 0, 'Target Population:', config_label_format)
    summary_sheet.write(row, 1, brand_config.get('target_population', 'Treated patients'))
    row += 1
    
    summary_sheet.write(row, 0, 'Adherence Rate:', config_label_format)
    summary_sheet.write(row, 1, brand_config.get('adherence', 0.68), rate_format)
    row += 1
    
    summary_sheet.write(row, 0, 'Annual Dose per Patient:', config_label_format)
    summary_sheet.write(row, 1, brand_config.get('annual_dose', 360), number_format)
    row += 1
    
    summary_sheet.write(row, 0, 'Price per Dose:', config_label_format)
    summary_sheet.write(row, 1, brand_config.get('price_per_dose', 16), currency_format)
    
    # ===== SHEET 2: EPIDEMIOLOGY FORECAST WITH FORMULAS =====
    epi_sheet = workbook.add_worksheet('Epidemiology Forecast')
    
    # Get data - filter out separator rows
    metrics = [m for m in epi_forecast.index if m.strip() and not m.startswith('===') and not m.startswith('  ')]
    years = list(epi_forecast.columns)
    
    base_pop = ai_assumptions.get('population', 0)
    base_year = ai_assumptions.get('base_year', datetime.now().year)
    growth_rates = ai_assumptions.get('growth_rates', {})
    known_values = ai_assumptions.get('known_values', {})
    derived_metrics = ai_assumptions.get('derived_metrics', {})
    
    # ===== ADD PARAMETERS SECTION =====
    epi_sheet.merge_range('A1:B1', 'MODEL PARAMETERS', header_format)
    epi_sheet.set_row(0, 20)
    
    # Base Population
    epi_sheet.write(1, 0, 'Base Population:', metric_name_format)
    epi_sheet.write(1, 1, base_pop, constant_format)
    base_pop_cell = 'B2'
    
    # Base Year
    epi_sheet.write(2, 0, 'Base Year:', metric_name_format)
    epi_sheet.write(2, 1, base_year, constant_format)
    
    # Population Growth Rate
    pop_growth_rate = growth_rates.get('Population', 0.007)
    epi_sheet.write(3, 0, 'Population Growth (annual):', metric_name_format)
    epi_sheet.write(3, 1, pop_growth_rate, constant_rate_format)
    pop_growth_cell = 'B4'
    
    # Base Rates
    epi_sheet.write(4, 0, 'Prevalence Rate:', metric_name_format)
    epi_sheet.write(4, 1, known_values.get('prevalence', 0.02), constant_rate_format)
    prev_rate_cell = 'B5'
    
    epi_sheet.write(5, 0, 'Diagnosed Rate:', metric_name_format)
    epi_sheet.write(5, 1, known_values.get('diagnosed', 0.85), constant_rate_format)
    diag_rate_cell = 'B6'
    
    epi_sheet.write(6, 0, 'Treated Rate:', metric_name_format)
    epi_sheet.write(6, 1, known_values.get('treated', 0.78), constant_rate_format)
    treat_rate_cell = 'B7'
    
    # Set column widths
    epi_sheet.set_column('A:A', 30)
    epi_sheet.set_column('B:B', 15)
    
    # ===== FORECAST TABLE STARTS AT ROW 9 =====
    forecast_start_row = 8
    
    # Write headers
    epi_sheet.write(forecast_start_row, 0, 'Metric', header_format)
    for col_idx, year in enumerate(years, start=1):
        epi_sheet.write(forecast_start_row, col_idx, str(year), header_format)
        epi_sheet.set_column(col_idx, col_idx, 15)
    
    # Track row positions
    epi_row_map = {}
    
    # Write metrics with formulas
    for row_idx, metric in enumerate(metrics, start=forecast_start_row + 1):
        epi_sheet.write(row_idx, 0, metric, metric_name_format)
        epi_row_map[metric] = row_idx + 1  # Excel row (1-indexed)
        
        # ===== CHECK FOR DERIVED/CUSTOM METRICS FIRST! =====
        # Try exact match first
        is_derived = metric in derived_metrics
        
        # If not found, try normalized lookup
        if not is_derived:
            metric_normalized = metric.strip()
            for dm_key in derived_metrics.keys():
                if dm_key.strip() == metric_normalized:
                    is_derived = True
                    # Use the actual key from derived_metrics for lookup
                    metric = dm_key
                    break
        
        if is_derived:
            # ===== CUSTOM/DERIVED METRICS - HANDLE FIRST =====
            config = derived_metrics[metric]
            source_metric = config.get('source')
            base_rate = config.get('base_rate', 0.0)
            is_rate_metric = config.get('is_rate', False)
            
            if is_rate_metric:
                # Custom rate - constant value
                for col_idx in range(1, len(years) + 1):
                    epi_sheet.write_number(row_idx, col_idx, base_rate, constant_rate_format)
            else:
                # Custom population - formula
                for col_idx in range(1, len(years) + 1):
                    col_letter = get_excel_column(col_idx)
                    
                    # Find source row
                    source_row = epi_row_map.get(source_metric)
                    
                    # Get rate metric from config (proper fix)
                    rate_metric_name = config.get('rate_metric')
                    
                    if not rate_metric_name:
                        # Legacy: construct rate name
                        if metric.endswith(' Population'):
                            base_name = metric[:-len(' Population')]
                            rate_metric_name = f"{base_name} Rate"
                        elif metric.endswith(' population'):
                            base_name = metric[:-len(' population')]
                            rate_metric_name = f"{base_name} Rate"
                    
                    # Find rate row
                    rate_row = epi_row_map.get(rate_metric_name) if rate_metric_name else None
                    
                    if source_row is not None and rate_row is not None:
                        # Formula: Source × Rate CELL (not constant)
                        formula = f'={col_letter}{source_row}*{col_letter}{rate_row}'
                        epi_sheet.write_formula(row_idx, col_idx, formula, number_format)
                    elif source_row is not None and base_rate > 0:
                        # No rate metric - use constant
                        formula = f'={col_letter}{source_row}*{base_rate}'
                        epi_sheet.write_formula(row_idx, col_idx, formula, number_format)
                    elif source_row is not None:
                        # No rate at all
                        formula = f'={col_letter}{source_row}'
                        epi_sheet.write_formula(row_idx, col_idx, formula, number_format)
                    else:
                        # Source not found - write actual value
                        try:
                            value = epi_forecast.loc[metric, years[col_idx-1]]
                            epi_sheet.write_number(row_idx, col_idx, float(value), number_format)
                        except:
                            epi_sheet.write_number(row_idx, col_idx, 0, number_format)
        
        else:
            # ===== STANDARD METRICS (only if not derived) =====
            metric_type, is_rate = classify_metric_for_excel(metric)
            
            # ===== TOTAL POPULATION =====
            if metric_type == 'total_population' and not is_rate:
                for col_idx, year in enumerate(years, start=1):
                    year_val = int(year) if str(year).isdigit() else base_year
                    years_from_base = year_val - base_year
                    
                    if col_idx == 1:
                        formula = f'={base_pop_cell}'
                        epi_sheet.write_formula(row_idx, col_idx, formula, constant_format)
                    else:
                        col_letter = get_excel_column(1)
                        formula = f'={col_letter}{row_idx + 1}*POWER(1+{pop_growth_cell},{years_from_base})'
                        epi_sheet.write_formula(row_idx, col_idx, formula, number_format)
            
            # ===== PREVALENCE RATE =====
            elif metric_type == 'prevalence' and is_rate:
                for col_idx in range(1, len(years) + 1):
                    formula = f'={prev_rate_cell}'
                    epi_sheet.write_formula(row_idx, col_idx, formula, constant_rate_format)
            
            # ===== PREVALENT POPULATION =====
            elif metric_type == 'prevalence' and not is_rate:
                for col_idx in range(1, len(years) + 1):
                    col_letter = get_excel_column(col_idx)
                    
                    total_pop_row = None
                    for m, r in epi_row_map.items():
                        m_type, m_is_rate = classify_metric_for_excel(m)
                        if m_type == 'total_population' and not m_is_rate:
                            total_pop_row = r
                            break
                    
                    prev_rate_row = None
                    for m, r in epi_row_map.items():
                        m_type, m_is_rate = classify_metric_for_excel(m)
                        if m_type == 'prevalence' and m_is_rate:
                            prev_rate_row = r
                            break
                    
                    if total_pop_row and prev_rate_row:
                        formula = f'={col_letter}{total_pop_row}*{col_letter}{prev_rate_row}'
                        epi_sheet.write_formula(row_idx, col_idx, formula, number_format)
                    else:
                        epi_sheet.write_number(row_idx, col_idx, 0, number_format)
            
            # ===== DIAGNOSED RATE =====
            elif metric_type == 'diagnosed' and is_rate:
                for col_idx in range(1, len(years) + 1):
                    formula = f'={diag_rate_cell}'
                    epi_sheet.write_formula(row_idx, col_idx, formula, constant_rate_format)
            
            # ===== DIAGNOSED POPULATION =====
            elif metric_type == 'diagnosed' and not is_rate:
                for col_idx in range(1, len(years) + 1):
                    col_letter = get_excel_column(col_idx)
                    
                    prev_pop_row = None
                    for m, r in epi_row_map.items():
                        m_type, m_is_rate = classify_metric_for_excel(m)
                        if m_type == 'prevalence' and not m_is_rate:
                            prev_pop_row = r
                            break
                    
                    diag_rate_row = None
                    for m, r in epi_row_map.items():
                        m_type, m_is_rate = classify_metric_for_excel(m)
                        if m_type == 'diagnosed' and m_is_rate:
                            diag_rate_row = r
                            break
                    
                    if prev_pop_row and diag_rate_row:
                        formula = f'={col_letter}{prev_pop_row}*{col_letter}{diag_rate_row}'
                        epi_sheet.write_formula(row_idx, col_idx, formula, number_format)
                    else:
                        epi_sheet.write_number(row_idx, col_idx, 0, number_format)
            
            # ===== TREATED RATE =====
            elif metric_type == 'treated' and is_rate:
                for col_idx in range(1, len(years) + 1):
                    formula = f'={treat_rate_cell}'
                    epi_sheet.write_formula(row_idx, col_idx, formula, constant_rate_format)
            
            # ===== TREATED POPULATION =====
            elif metric_type == 'treated' and not is_rate:
                for col_idx in range(1, len(years) + 1):
                    col_letter = get_excel_column(col_idx)
                    
                    diag_pop_row = None
                    for m, r in epi_row_map.items():
                        m_type, m_is_rate = classify_metric_for_excel(m)
                        if m_type == 'diagnosed' and not m_is_rate:
                            diag_pop_row = r
                            break
                    
                    treat_rate_row = None
                    for m, r in epi_row_map.items():
                        m_type, m_is_rate = classify_metric_for_excel(m)
                        if m_type == 'treated' and m_is_rate:
                            treat_rate_row = r
                            break
                    
                    if diag_pop_row and treat_rate_row:
                        formula = f'={col_letter}{diag_pop_row}*{col_letter}{treat_rate_row}'
                        epi_sheet.write_formula(row_idx, col_idx, formula, number_format)
                    else:
                        epi_sheet.write_number(row_idx, col_idx, 0, number_format)
            
            # ===== FALLBACK =====
            else:
                for col_idx, year in enumerate(years, start=1):
                    try:
                        value = epi_forecast.loc[metric, year]
                        if is_rate:
                            epi_sheet.write_number(row_idx, col_idx, float(value), rate_format)
                        else:
                            epi_sheet.write_number(row_idx, col_idx, float(value), number_format)
                    except:
                        epi_sheet.write_number(row_idx, col_idx, 0, number_format)    
    # ===== SHEET 3: UPTAKE CURVES =====
    uptake_sheet = workbook.add_worksheet('Uptake Curves')
    
    uptake_sheet.merge_range(0, 0, 0, 15, 'ANALOGUE UPTAKE SHARES (STANDARD CURVES)', title_format)
    uptake_sheet.set_row(0, 20)
    
    uptake_sheet.write(2, 0, '', header_format)
    uptake_sheet.set_column('A:A', 20)
    
    for col_idx in range(15):
        uptake_sheet.write(1, col_idx + 1, 2025 + col_idx, header_format)
        uptake_sheet.write(2, col_idx + 1, f'Y{col_idx + 1}', header_format)
        uptake_sheet.set_column(col_idx + 1, col_idx + 1, 10)
    
    curve_row_map = {}
    for row_idx, (curve_name, curve_values) in enumerate(UPTAKE_CURVES.items(), start=3):
        uptake_sheet.write(row_idx, 0, curve_name, metric_name_format)
        curve_row_map[curve_name] = row_idx + 1
        
        for col_idx, value in enumerate(curve_values, start=1):
            uptake_sheet.write(row_idx, col_idx, value, rate_format)
    
    # ===== SHEET 4: BRAND FORECAST WITH FORMULAS =====
    brand_sheet = workbook.add_worksheet('Brand Forecast')
    
    brand_sheet.merge_range(0, 0, 0, len(years), f"{brand_config.get('brand_name', 'Brand X').upper()} FORECAST", title_format)
    brand_sheet.set_row(0, 20)
    
    # Configuration section
    config_start_row = 2
    brand_sheet.merge_range(config_start_row, 0, config_start_row, 1, 'BRAND CONFIGURATION', section_header_format)
    
    config_row = config_start_row + 1
    brand_sheet.write(config_row, 0, 'Launch Year:', config_label_format)
    brand_sheet.write(config_row, 1, brand_config.get('launch_year', 2025), constant_format)
    launch_year_cell = f'B{config_row + 1}'
    
    config_row += 1
    brand_sheet.write(config_row, 0, 'Peak Share:', config_label_format)
    brand_sheet.write(config_row, 1, brand_config.get('peak_share', 0.35), constant_rate_format)
    peak_share_cell = f'B{config_row + 1}'
    
    config_row += 1
    brand_sheet.write(config_row, 0, 'Uptake Type:', config_label_format)
    brand_sheet.write(config_row, 1, brand_config.get('uptake_type', '7 Year Medium'))
    
    config_row += 1
    brand_sheet.write(config_row, 0, 'Adherence:', config_label_format)
    brand_sheet.write(config_row, 1, brand_config.get('adherence', 0.68), constant_rate_format)
    adherence_cell = f'B{config_row + 1}'
    
    config_row += 1
    brand_sheet.write(config_row, 0, 'Annual Dose per Patient:', config_label_format)
    brand_sheet.write(config_row, 1, brand_config.get('annual_dose', 360), constant_format)
    annual_dose_cell = f'B{config_row + 1}'
    
    config_row += 1
    brand_sheet.write(config_row, 0, 'Price per Dose ($):', config_label_format)
    brand_sheet.write(config_row, 1, brand_config.get('price_per_dose', 16), constant_currency_format)
    price_cell = f'B{config_row + 1}'
    
    brand_sheet.set_column('A:A', 35)
    brand_sheet.set_column('B:B', 15)
    
    # Forecast table
    forecast_start_row = config_row + 3
    launch_year = brand_config.get('launch_year', 2025)
    uptake_type = brand_config.get('uptake_type', '7 Year Medium')
    uptake_curve_row = curve_row_map.get(uptake_type, 4)
    target_population = brand_config.get('target_population', 'Treated patients')
    
    # Find target population row in epi forecast
    target_pop_excel_row = None
    for metric, row_num in epi_row_map.items():
        if target_population.lower() in metric.lower():
            target_pop_excel_row = row_num
            break
    
    # Headers
    brand_sheet.write(forecast_start_row, 0, 'Metric', brand_header_format)
    for col_idx, year in enumerate(years, start=1):
        brand_sheet.write(forecast_start_row, col_idx, str(year), brand_header_format)
        brand_sheet.set_column(col_idx, col_idx, 15)
    
    brand_row_map = {}
    current_row = forecast_start_row + 1
    
    # Brand X Patient Share
    metric = 'Brand X Patient Share'
    brand_row_map[metric] = current_row + 1
    brand_sheet.write(current_row, 0, metric, metric_name_format)
    
    for col_idx, year in enumerate(years, start=1):
        try:
            year_int = int(year)
        except:
            year_int = 2025
        
        years_since_launch = year_int - launch_year
        col_letter = get_excel_column(col_idx)
        
        if years_since_launch < 0:
            brand_sheet.write(current_row, col_idx, 0, rate_format)
        elif years_since_launch >= 15:
            formula = f'={peak_share_cell}*1'
            brand_sheet.write_formula(current_row, col_idx, formula, rate_format)
        else:
            uptake_col = years_since_launch + 2
            uptake_col_letter = get_excel_column(uptake_col)
            formula = f'={peak_share_cell}*\'Uptake Curves\'!{uptake_col_letter}${uptake_curve_row}'
            brand_sheet.write_formula(current_row, col_idx, formula, rate_format)
    
    current_row += 2
    
    # Patients on Brand X
    metric = 'Patients on Brand X'
    brand_row_map[metric] = current_row + 1
    brand_sheet.write(current_row, 0, metric, metric_name_format)
    
    brand_share_row = brand_row_map['Brand X Patient Share']
    
    for col_idx in range(1, len(years) + 1):
        col_letter = get_excel_column(col_idx)
        
        if target_pop_excel_row:
            formula = f'=\'Epidemiology Forecast\'!{col_letter}${target_pop_excel_row}*{col_letter}${brand_share_row}'
            brand_sheet.write_formula(current_row, col_idx, formula, number_format)
        else:
            brand_sheet.write(current_row, col_idx, 0, number_format)
    
    current_row += 2
    
    # Adherence
    metric = 'Adherence'
    brand_row_map[metric] = current_row + 1
    brand_sheet.write(current_row, 0, metric, metric_name_format)
    
    for col_idx in range(1, len(years) + 1):
        formula = f'={adherence_cell}'
        brand_sheet.write_formula(current_row, col_idx, formula, constant_rate_format)
    
    current_row += 1
    
    # Adherent Patients
    metric = 'Adherent Patients'
    brand_row_map[metric] = current_row + 1
    brand_sheet.write(current_row, 0, metric, metric_name_format)
    
    patients_row = brand_row_map['Patients on Brand X']
    adherence_row = brand_row_map['Adherence']
    
    for col_idx in range(1, len(years) + 1):
        col_letter = get_excel_column(col_idx)
        formula = f'={col_letter}${patients_row}*{col_letter}${adherence_row}'
        brand_sheet.write_formula(current_row, col_idx, formula, number_format)
    
    current_row += 2
    
    # Annual dose per patient
    metric = 'Annual dose per patient'
    brand_row_map[metric] = current_row + 1
    brand_sheet.write(current_row, 0, metric, metric_name_format)
    
    for col_idx in range(1, len(years) + 1):
        formula = f'={annual_dose_cell}'
        brand_sheet.write_formula(current_row, col_idx, formula, constant_format)
    
    current_row += 1
    
    # Price per dose
    metric = 'Price per dose ($)'
    brand_row_map[metric] = current_row + 1
    brand_sheet.write(current_row, 0, metric, metric_name_format)
    
    for col_idx in range(1, len(years) + 1):
        formula = f'={price_cell}'
        brand_sheet.write_formula(current_row, col_idx, formula, constant_currency_format)
    
    current_row += 1
    
    # Total adherence adjusted Volume
    metric = 'Total adherence adjusted Volume'
    brand_row_map[metric] = current_row + 1
    brand_sheet.write(current_row, 0, metric, metric_name_format)
    
    adherent_row = brand_row_map['Adherent Patients']
    dose_row = brand_row_map['Annual dose per patient']
    
    for col_idx in range(1, len(years) + 1):
        col_letter = get_excel_column(col_idx)
        formula = f'={col_letter}${adherent_row}*{col_letter}${dose_row}'
        brand_sheet.write_formula(current_row, col_idx, formula, number_format)
    
    current_row += 2
    
    # Revenue in billions
    metric = 'Revenue in billions'
    brand_row_map[metric] = current_row + 1
    brand_sheet.write(current_row, 0, metric, metric_name_format)
    
    volume_row = brand_row_map['Total adherence adjusted Volume']
    price_row = brand_row_map['Price per dose ($)']
    
    for col_idx in range(1, len(years) + 1):
        col_letter = get_excel_column(col_idx)
        formula = f'={col_letter}${volume_row}*{col_letter}${price_row}/1000000000'
        brand_sheet.write_formula(current_row, col_idx, formula, revenue_format)
    
    # ===== SHEET 5: INSTRUCTIONS =====
    instructions_sheet = workbook.add_worksheet('Instructions')
    instructions_sheet.set_column('A:A', 80)
    
    instructions = [
        ['BRAND FORECAST MODEL - USER GUIDE', ''],
        ['', ''],
        ['OVERVIEW', ''],
        ['This workbook contains a comprehensive brand forecast model that links epidemiology data with commercial projections.', ''],
        ['ALL epidemiology and brand calculations use FORMULAS - you can modify parameters and see instant updates!', ''],
        ['', ''],
        ['SHEETS EXPLAINED', ''],
        ['', ''],
        ['1. Summary', 'High-level overview of disease parameters and brand configuration'],
        ['2. Epidemiology Forecast', 'Patient flow with FORMULAS - modify gray parameter cells at top'],
        ['3. Uptake Curves', 'Standard market adoption curves for brand uptake modeling'],
        ['4. Brand Forecast', 'Complete brand forecast with FORMULAS - modify gray config cells'],
        ['5. Instructions', 'This sheet - how to use the model'],
        ['', ''],
        ['HOW TO USE - EPIDEMIOLOGY FORECAST', ''],
        ['', ''],
        ['Go to "Epidemiology Forecast" sheet:', ''],
        ['- GRAY CELLS at top (B2:B7) are editable parameters', ''],
        ['- Change Base Population, Growth Rate, or any Rate (Prevalence/Diagnosed/Treated)', ''],
        ['- ALL population metrics recalculate automatically via formulas', ''],
        ['- Formula structure: Total Pop → Prevalence → Diagnosed → Treated', ''],
        ['', ''],
        ['HOW TO USE - BRAND FORECAST', ''],
        ['', ''],
        ['Go to "Brand Forecast" sheet:', ''],
        ['- GRAY CELLS (B4:B9) are editable brand parameters', ''],
        ['- Launch Year: When brand enters market', ''],
        ['- Peak Share: Maximum achievable market share', ''],
        ['- Adherence: % of patients staying on treatment', ''],
        ['- Annual Dose: Doses per patient per year', ''],
        ['- Price per Dose: Price in dollars', ''],
        ['', ''],
        ['FORMULAS EXPLAINED', ''],
        ['', ''],
        ['Epidemiology Formulas:', ''],
        ['Total Population = Base Pop × (1 + Growth)^years', ''],
        ['Prevalent Pop = Total Pop × Prevalence Rate', ''],
        ['Diagnosed Pop = Prevalent Pop × Diagnosed Rate', ''],
        ['Treated Pop = Diagnosed Pop × Treated Rate', ''],
        ['', ''],
        ['Brand Formulas:', ''],
        ['Brand Share = Peak Share × Uptake Curve %', ''],
        ['Patients on Brand = Treated Pop × Brand Share', ''],
        ['Adherent Patients = Patients on Brand × Adherence', ''],
        ['Total Volume = Adherent Patients × Annual Dose', ''],
        ['Revenue = (Volume × Price) / 1,000,000,000', ''],
        ['', ''],
        ['SCENARIO MODELING', ''],
        ['', ''],
        ['Try these by changing gray parameter cells:', ''],
        ['', ''],
        ['Epidemiology Scenarios:', ''],
        ['- Higher prevalence: Change Prevalence Rate from 0.36 to 0.40', ''],
        ['- Better diagnosis: Change Diagnosed Rate from 0.85 to 0.90', ''],
        ['- More treatment: Change Treated Rate from 0.78 to 0.85', ''],
        ['', ''],
        ['Brand Scenarios:', ''],
        ['- Premium pricing: Change Price from $16 to $20', ''],
        ['- Market leader: Change Peak Share from 35% to 45%', ''],
        ['- Better adherence: Change Adherence from 68% to 75%', ''],
        ['- Faster uptake: Change Uptake Type (manually select different curve)', ''],
        ['', ''],
        ['ALL changes automatically update entire model!', ''],
        ['', ''],
        [f'Model generated: {datetime.now().strftime("%Y-%m-%d %H:%M:%S")}', '']
    ]
    
    for row_idx, (col1, col2) in enumerate(instructions):
        if any(keyword in col1 for keyword in ['GUIDE', 'OVERVIEW', 'SHEETS EXPLAINED', 'HOW TO USE', 'FORMULAS EXPLAINED', 'SCENARIO MODELING']):
            instructions_sheet.write(row_idx, 0, col1, title_format)
        else:
            instructions_sheet.write(row_idx, 0, col1)
        
        if col2:
            instructions_sheet.write(row_idx, 1, col2)
    
    # ===== SHEET 6: CITATIONS =====
    citations = ai_assumptions.get('citations', {})
    if citations:
        citations_sheet = workbook.add_worksheet('Research Citations')
        
        citations_sheet.merge_range(0, 0, 0, 2, 'RESEARCH CITATIONS & DATA SOURCES', title_format)
        citations_sheet.set_row(0, 20)
        
        citations_sheet.write(2, 0, 'Metric', header_format)
        citations_sheet.write(2, 1, 'Value', header_format)
        citations_sheet.write(2, 2, 'Source', header_format)
        
        citations_sheet.set_column('A:A', 25)
        citations_sheet.set_column('B:B', 15)
        citations_sheet.set_column('C:C', 80)
        
        row = 3
        known_values = ai_assumptions.get('known_values', {})
        
        for metric, citation in citations.items():
            metric_display = metric.replace('_', ' ').title()
            citations_sheet.write(row, 0, metric_display, metric_name_format)
            
            value = known_values.get(metric)
            if value:
                citations_sheet.write(row, 1, value, rate_format)
            else:
                citations_sheet.write(row, 1, 'N/A')
            
            citations_sheet.write(row, 2, citation, metric_name_format)
            row += 1
    
    # ===== SHEET 7: EVENTS (if any) =====
    if events := brand_config.get('events'):
        events_sheet = workbook.add_worksheet('Market Events')
        
        events_sheet.merge_range(0, 0, 0, 5, 'MARKET EVENTS TIMELINE', title_format)
        events_sheet.set_row(0, 20)
        
        headers = ['Year', 'Event Type', 'Description', 'Post-Event Peak', 'Years to Peak', 'Curve']
        for col_idx, header in enumerate(headers):
            events_sheet.write(2, col_idx, header, header_format)
        
        for row_idx, event in enumerate(events, start=3):
            events_sheet.write(row_idx, 0, event['event_year'])
            events_sheet.write(row_idx, 1, event['event_type'])
            events_sheet.write(row_idx, 2, event.get('description', ''))
            events_sheet.write(row_idx, 3, event['post_event_peak_share'], rate_format)
            events_sheet.write(row_idx, 4, event['years_to_peak'])
            events_sheet.write(row_idx, 5, event['curve_type'])
        
        events_sheet.set_column('A:A', 10)
        events_sheet.set_column('B:B', 25)
        events_sheet.set_column('C:C', 40)
        events_sheet.set_column('D:D', 15)
        events_sheet.set_column('E:E', 15)
        events_sheet.set_column('F:F', 25)
    
    writer.close()
    
    with open(filename, 'rb') as f:
        return f.read()
    
def apply_custom_css():
    """Apply custom CSS for modern, professional UI design"""
    st.markdown("""
    <style>
    /* ===== GLOBAL STYLES ===== */
    :root {
        --bg: #f3f6ff;
        --card: #ffffff;
        --text: #0f172a;
        --muted: #475569;
        --primary: #0b69d6;
        --primary-2: #0e4db6;
        --accent: #10b981;
        --border: rgba(148, 163, 184, 0.6);
        --shadow: rgba(15, 23, 42, 0.08);
    }

    * {
        scroll-behavior: smooth;
    }

    body, .main, .stApp {
        background: var(--bg);
        color: var(--text);
        font-family: "Segoe UI", Roboto, Helvetica, Arial, sans-serif;
        /* Increase base font size by ~2pt for better legibility */
        font-size: 1.15rem;
    }

    table, th, td {
        /* Ensure table text (especially numeric values) is larger and easier to read */
        font-size: 1.15rem !important;
    }

    .main {
        padding: 1.2rem 2rem;
    }

    /* ===== HEADER & BRANDING ===== */
    .app-header {
        background: linear-gradient(135deg, var(--primary) 0%, var(--primary-2) 100%);
        padding: 2rem;
        border-radius: 14px;
        margin-bottom: 2rem;
        box-shadow: 0 12px 25px rgba(15, 23, 42, 0.12);
        color: white;
    }

    .app-header h1 {
        color: white;
        margin: 0;
        font-size: 2.6rem;
        font-weight: 800;
        letter-spacing: -0.02em;
    }

    .app-header p {
        color: rgba(255, 255, 255, 0.85);
        margin: 0.6rem 0 0 0;
        font-size: 1.15rem;
        line-height: 1.5;
    }

    /* ===== CARDS & CONTAINERS ===== */
    .info-card {
        background: #f0f7ff;
        border-left: 4px solid rgb(14, 77, 182);
        padding: 1.5rem;
        border-radius: 12px;
        margin: 1rem 0;
        box-shadow: 0 8px 18px rgba(15, 23, 42, 0.08);
    }

    .success-card {
        background: #ecfdf5;
        border-left: 4px solid rgb(16, 185, 129);
        padding: 1.5rem;
        border-radius: 12px;
        margin: 1rem 0;
        box-shadow: 0 8px 18px rgba(15, 23, 42, 0.06);
    }

    .warning-card {
        background: #fff7ed;
        border-left: 4px solid rgb(245, 158, 11);
        padding: 1.5rem;
        border-radius: 12px;
        margin: 1rem 0;
        box-shadow: 0 8px 18px rgba(15, 23, 42, 0.06);
    }

    /* ===== BUTTONS ===== */
    .stButton > button {
        border-radius: 12px;
        padding: 1rem 2.2rem;
        font-weight: 700;
        font-size: 17px;
        transition: transform 0.18s ease, box-shadow 0.18s ease;
        border: none;
        box-shadow: 0 8px 22px rgba(15, 23, 42, 0.14);
        color: #0f172a;
        background: linear-gradient(135deg, rgba(56, 189, 248, 0.9) 0%, rgba(96, 165, 250, 0.9) 100%);
    }

    .stButton > button:hover {
        transform: translateY(-2px);
        box-shadow: 0 12px 28px rgba(15, 23, 42, 0.2);
        background: linear-gradient(135deg, rgba(56, 189, 248, 1) 0%, rgba(37, 99, 235, 1) 100%);
    }

    /* Ensure consistent button styling regardless of kind */
    .stButton > button[kind="primary"],
    .stButton > button[kind="secondary"] {
        background: linear-gradient(135deg, rgba(56, 189, 248, 0.9) 0%, rgba(96, 165, 250, 0.9) 100%);
        color: #0f172a;
    }

    .stButton > button[kind="primary"]:hover,
    .stButton > button[kind="secondary"]:hover {
        background: linear-gradient(135deg, rgba(56, 189, 248, 1) 0%, rgba(37, 99, 235, 1) 100%);
    }

    /* ===== INPUTS ===== */
    .stTextInput > div > div > input,
    .stTextArea > div > div > textarea,
    .stNumberInput > div > div > input,
    .stSelectbox > div > div > select,
    .stMultiSelect > div > div > div > div > input {
        border-radius: 12px;
        border: 1px solid var(--border);
        transition: border-color 0.25s ease, box-shadow 0.25s ease;
        background-color: rgba(255, 255, 255, 0.9);
        padding: 0.75rem 0.9rem;
        font-size: 16px;
        font-family: inherit;
        color: var(--text);
        line-height: 1.45;
        box-shadow: inset 0 2px 6px rgba(15, 23, 42, 0.06);
        min-height: 50px;
    }

    .stTextInput > div > div > input::placeholder,
    .stTextArea > div > div > textarea::placeholder,
    .stNumberInput > div > div > input::placeholder,
    .stSelectbox > div > div > select::placeholder,
    .stMultiSelect > div > div > div > div > input::placeholder {
        color: rgba(55, 65, 81, 0.5);
        font-weight: 500;
    }

    .stTextInput > div > div > input:focus,
    .stTextArea > div > div > textarea:focus,
    .stNumberInput > div > div > input:focus,
    .stSelectbox > div > div > select:focus,
    .stMultiSelect > div > div > div > div > input:focus {
        border-color: var(--primary);
        box-shadow: 0 0 0 3px rgba(11, 105, 214, 0.18);
        outline: none;
        background-color: white;
    }

    /* ===== TABS ===== */
    .stTabs [data-baseweb="tab-list"] {
        gap: 10px;
        background-color: rgba(15, 23, 42, 0.04);
        padding: 0.6rem;
        border-radius: 10px;
    }

    .stTabs [data-baseweb="tab"] {
        border-radius: 10px;
        padding: 0.75rem 1.5rem;
        font-weight: 600;
        transition: all 0.18s ease;
    }

    .stTabs [aria-selected="true"] {
        background: linear-gradient(135deg, var(--primary) 0%, var(--primary-2) 100%);
        color: white;
    }

    /* ===== EXPANDERS ===== */
    .streamlit-expanderHeader {
        border-radius: 10px;
        background-color: rgba(15, 23, 42, 0.06);
        font-weight: 600;
        transition: background-color 0.25s ease;
    }

    .streamlit-expanderHeader:hover {
        background-color: rgba(15, 23, 42, 0.1);
    }

    /* ===== METRICS ===== */
    [data-testid="stMetricValue"] {
        font-size: 2.2rem;
        font-weight: 800;
        color: var(--primary);
    }

    /* ===== DATAFRAMES ===== */
    .dataframe {
        border-radius: 12px;
        overflow: hidden;
        background-color: var(--card);
        box-shadow: 0 8px 20px rgba(15, 23, 42, 0.06);
    }

    .dataframe thead tr th {
        background: linear-gradient(135deg, var(--primary) 0%, var(--primary-2) 100%);
        color: white;
        font-weight: 600;
        padding: 1rem;
    }

    .dataframe tbody tr:nth-child(even) {
        background-color: rgba(14, 77, 182, 0.05);
    }

    .dataframe tbody tr:hover {
        background-color: rgba(14, 77, 182, 0.1);
    }

    /* ===== SIDEBAR ===== */
    [data-testid="stSidebar"] {
        background: linear-gradient(180deg, rgba(11, 105, 214, 0.06) 0%, rgba(14, 77, 182, 0.12) 100%);
    }

    [data-testid="stSidebar"] .stMarkdown {
        padding: 0.5rem 0;
    }

    /* ===== PROGRESS INDICATORS ===== */
    .progress-step {
        display: flex;
        align-items: center;
        padding: 0.75rem;
        margin: 0.5rem 0;
        border-radius: 12px;
        background-color: rgba(15, 23, 42, 0.04);
        transition: all 0.3s ease;
    }

    .progress-step.completed {
        background: linear-gradient(135deg, rgba(16, 185, 129, 0.15) 0%, rgba(16, 185, 129, 0.08) 100%);
        border-left: 4px solid var(--accent);
    }

    .progress-step.active {
        background: linear-gradient(135deg, rgba(14, 77, 182, 0.15) 0%, rgba(14, 77, 182, 0.08) 100%);
        border-left: 4px solid var(--primary);
    }

    /* ===== TOOLTIPS & HELP ===== */
    .tooltip-icon {
        display: inline-block;
        width: 18px;
        height: 18px;
        background-color: var(--primary);
        color: white;
        border-radius: 50%;
        text-align: center;
        font-size: 12px;
        line-height: 18px;
        cursor: help;
        margin-left: 4px;
    }

    /* ===== LOADING ANIMATIONS ===== */
    @keyframes pulse {
        0%, 100% { opacity: 1; }
        50% { opacity: 0.4; }
    }

    .loading-indicator {
        animation: pulse 2s cubic-bezier(0.4, 0, 0.6, 1) infinite;
    }

    /* ===== SECTION HEADERS ===== */
    .section-header {
        display: flex;
        align-items: center;
        gap: 0.75rem;
        padding: 1rem 0;
        border-bottom: 3px solid rgba(14, 77, 182, 0.3);
        margin-bottom: 1.5rem;
    }

    .section-header h2 {
        margin: 0;
        color: var(--primary);
    }

    .section-icon {
        font-size: 1.5rem;
    }

    /* ===== QUICK START GUIDE ===== */
    .quick-start {
        background: linear-gradient(135deg, rgba(255, 248, 225, 1) 0%, rgba(255, 243, 205, 1) 100%);
        border-radius: 12px;
        padding: 1.5rem;
        margin: 1rem 0;
        border-left: 4px solid rgba(245, 158, 11, 1);
    }

    .quick-start h3 {
        color: rgba(134, 46, 0, 0.9);
        margin-top: 0;
    }

    .quick-start ol {
        color: rgba(71, 85, 105, 0.9);
        font-weight: 500;
    }

    /* ===== DARK MODE SUPPORT ===== */
    @media (prefers-color-scheme: dark) {
        body, .main, .stApp {
            background: #0b1220;
            color: rgba(226, 232, 240, 0.9);
        }

        .app-header {
            background: linear-gradient(135deg, #0e1a2b 0%, #1a2b44 100%);
        }

        .stTextInput > div > div > input,
        .stTextArea > div > div > textarea,
        .stNumberInput > div > div > input,
        .stSelectbox > div > div > select,
        .stMultiSelect > div > div > div > div > input {
            background-color: rgba(15, 23, 42, 0.75);
            border-color: rgba(148, 163, 184, 0.4);
            color: rgba(226, 232, 240, 0.9);
        }

        .stTextInput > div > div > input::placeholder,
        .stTextArea > div > div > textarea::placeholder,
        .stNumberInput > div > div > input::placeholder,
        .stSelectbox > div > div > select::placeholder,
        .stMultiSelect > div > div > div > div > input::placeholder {
            color: rgba(148, 163, 184, 0.7);
        }

        .dataframe {
            background-color: rgba(15, 23, 42, 0.8);
            box-shadow: none;
        }

        .dataframe thead tr th {
            background: rgba(11, 105, 214, 0.9);
        }

        .dataframe tbody tr:nth-child(even) {
            background-color: rgba(11, 105, 214, 0.1);
        }

        .section-header {
            border-bottom-color: rgba(11, 105, 214, 0.4);
        }

        .section-header h2 {
            color: rgba(96, 165, 250, 0.9);
        }
    }
    </style>
    """, unsafe_allow_html=True)

def show_welcome_screen():
    """Display welcome screen with quick start guide"""
    if 'show_welcome' not in st.session_state:
        st.session_state['show_welcome'] = True
    
    if st.session_state.get('show_welcome', True):
        st.markdown("""
        <div class="quick-start">
            <h3>🚀 Quick Start Guide</h3>
            <p><strong>Welcome to the AI-Driven Epidemiology Forecast Application!</strong></p>
            <ol>
                <li><strong>Enter Disease Information:</strong> Paste or type your structured input with disease details</li>
                <li><strong>Configure Timeframe:</strong> Set frequency (Yearly/Quarterly/Monthly) and year range</li>
                <li><strong>Parse Input:</strong> Click "Parse & Continue" to process your data</li>
                <li><strong>Generate Base Forecast:</strong> Use AI Sub-Agent or Test Data to create initial forecast</li>
                <li><strong>Customize (Optional):</strong> Add custom metrics or modify parameters</li>
                <li><strong>Brand Forecast:</strong> Add brand-specific parameters for market analysis</li>
                <li><strong>Export & Compare:</strong> Download Excel models and compare scenarios</li>
            </ol>
            <p><em>💡 Tip: Use the "Test with Sample Data" button for a quick demo!</em></p>
        </div>
        """, unsafe_allow_html=True)
        
        col1, col2 = st.columns([3, 1])
        with col2:
            if st.button("Got it! Let's start", use_container_width=True):
                st.session_state['show_welcome'] = False
                st.rerun()

def show_progress_indicator():
    """Show workflow progress indicator in sidebar"""
    st.sidebar.markdown("### 📊 Workflow Progress")
    
    steps = [
        ("📝 Input Data", 'parsed' in st.session_state),
        ("🤖 AI Analysis", 'ai_assumptions_v2' in st.session_state),
        ("📈 Base Forecast", 'df_metrics_base' in st.session_state or 'df_metrics_custom' in st.session_state),
        ("💊 Brand Forecast", 'df_brand_forecast' in st.session_state),
        ("📊 Sensitivity Analysis", 'df_brand_forecast' in st.session_state),
        ("🔄 Scenarios", 'saved_scenarios' in st.session_state and len(st.session_state['saved_scenarios']) > 0)
    ]
    
    for step_name, is_completed in steps:
        status = "✅" if is_completed else "⏳"
        color = "#10b981" if is_completed else "#6b7280"
        st.sidebar.markdown(
            f"<div style='padding: 0.5rem; margin: 0.25rem 0; border-left: 3px solid {color}; background-color: rgba(59, 130, 246, 0.05); border-radius: 4px;'>"
            f"{status} {step_name}"
            f"</div>",
            unsafe_allow_html=True
        )

# Replace the apply_market_events function (around line 2700)

def apply_market_events(df_forecast: pd.DataFrame, events: list, start_year: int) -> pd.DataFrame:
    """Apply market events that affect epidemiology forecast metrics
    
    Args:
        df_forecast: Base epidemiology forecast DataFrame
        events: List of market event dictionaries
        start_year: Forecast start year
    
    Returns:
        Modified forecast DataFrame with events applied
    """
    if not events:
        return df_forecast
    
    df_modified = df_forecast.copy()
    
    # FIX: Ensure columns are strings for consistent access
    df_modified.columns = df_modified.columns.astype(str)
    
    # Get years as a list - keep them as strings
    years_str = [str(col) for col in df_modified.columns if str(col).isdigit()]
    
    # Sort events by year
    events_sorted = sorted(events, key=lambda x: x['event_year'])
    
    for event in events_sorted:
        event_year = event['event_year']
        target_metric = event['target_metric']
        impact_type = event['impact_type']  # 'percentage_change' or 'absolute_change'
        impact_value = event['impact_value']
        
        if target_metric not in df_modified.index:
            continue
        
        # Apply impact to all years from event year onwards
        for year_str in years_str:
            try:
                # Convert string year to int for comparison
                year_int = int(year_str)
                
                if year_int >= event_year:
                    # FIX: Access using string column name
                    current_value = df_modified.loc[target_metric, year_str]
                    
                    if impact_type == 'percentage_change':
                        # Apply percentage change (e.g., -10% means multiply by 0.9)
                        new_value = current_value * (1 + impact_value)
                    elif impact_type == 'absolute_change':
                        # Apply absolute change
                        new_value = current_value + impact_value
                    else:
                        continue
                    
                    # Update the value using string column name
                    df_modified.loc[target_metric, year_str] = new_value
                    
                    # Ensure non-negative values for population metrics
                    if 'population' in target_metric.lower() and 'rate' not in target_metric.lower():
                        if new_value < 0:
                            df_modified.loc[target_metric, year_str] = 0
                            
            except ValueError:
                # Skip non-numeric column names
                continue
            except Exception as e:
                # Log the error but continue processing
                import streamlit as st
                st.warning(f"Warning: Could not process year {year_str} for {target_metric}: {str(e)}")
                continue
    
    return df_modified


# Find the export_complete_model_with_events function around line 2750
# Replace the ENTIRE function with this fixed version:

# Find the export_complete_model_with_events function around line 2750
# Replace the ENTIRE function with this updated version:

def export_complete_model_with_events(epi_forecast: pd.DataFrame, brand_forecast: pd.DataFrame,
                                     brand_config: Dict, ai_assumptions: Dict, 
                                     market_events: list = None, brand_events: list = None,
                                     filename: str = None) -> bytes:
    """Export comprehensive model including both market and brand events with dedicated events forecast sheet
    
    This export includes:
    - Sheet 1: Forecast with Events (main results) - WITH FORMULAS
    - Sheet 2: Forecast without Events (baseline comparison)
    - Sheet 3: Summary
    - Sheet 4: Market Events Detail
    - Sheet 5: Brand Events Detail
    - Sheet 6: Instructions
    - Sheet 7: Research Citations
    """
    
    if filename is None:
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        filename = f"forecast_with_events_{timestamp}.xlsx"
    
    # FIX: Ensure both dataframes have string columns FIRST
    epi_forecast = epi_forecast.copy()
    epi_forecast.columns = epi_forecast.columns.astype(str)
    
    brand_forecast = brand_forecast.copy()
    brand_forecast.columns = brand_forecast.columns.astype(str)
    
    # Get original forecast without events from session state
    # This is passed in from the calling function
    import streamlit as st
    if 'df_metrics_custom' in st.session_state:
        epi_original = st.session_state['df_metrics_custom'].copy()
    else:
        epi_original = st.session_state.get('df_metrics_base', epi_forecast).copy()
    
    epi_original.columns = epi_original.columns.astype(str)
    
    brand_original = st.session_state.get('df_brand_forecast', brand_forecast).copy()
    brand_original.columns = brand_original.columns.astype(str)
    
    # Get derived metrics info for custom metrics
    derived_metrics = ai_assumptions.get('derived_metrics', {})
    base_pop = ai_assumptions.get('population', 0)
    base_year = ai_assumptions.get('base_year', datetime.now().year)
    growth_rates = ai_assumptions.get('growth_rates', {})
    known_values = ai_assumptions.get('known_values', {})
    
    writer = pd.ExcelWriter(filename, engine='xlsxwriter')
    workbook = writer.book
    
    # ===== DEFINE FORMATS =====
    title_format = workbook.add_format({
        'bold': True,
        'font_size': 14,
        'bg_color': '#2E75B6',
        'font_color': 'white',
        'align': 'center',
        'valign': 'vcenter',
        'border': 1
    })
    
    header_format = workbook.add_format({
        'bold': True,
        'bg_color': '#4472C4',
        'font_color': 'white',
        'border': 1,
        'align': 'center',
        'valign': 'vcenter',
        'text_wrap': True
    })
    
    brand_header_format = workbook.add_format({
        'bold': True,
        'bg_color': '#70AD47',
        'font_color': 'white',
        'border': 1,
        'align': 'center',
        'valign': 'vcenter',
        'text_wrap': True
    })
    
    event_header_format = workbook.add_format({
        'bold': True,
        'bg_color': '#FF6B6B',
        'font_color': 'white',
        'border': 1,
        'align': 'center',
        'valign': 'vcenter',
        'text_wrap': True
    })
    
    original_header_format = workbook.add_format({
        'bold': True,
        'bg_color': '#808080',
        'font_color': 'white',
        'border': 1,
        'align': 'center',
        'valign': 'vcenter',
        'text_wrap': True
    })
    
    config_label_format = workbook.add_format({
        'bold': True,
        'border': 1,
        'bg_color': '#D9E1F2',
        'align': 'right'
    })
    
    constant_format = workbook.add_format({
        'num_format': '#,##0',
        'border': 1,
        'bg_color': '#E7E6E6'
    })
    
    constant_rate_format = workbook.add_format({
        'num_format': '0.00%',
        'border': 1,
        'bg_color': '#E7E6E6'
    })
    
    number_format = workbook.add_format({
        'num_format': '#,##0',
        'border': 1
    })
    
    rate_format = workbook.add_format({
        'num_format': '0.00%',
        'border': 1
    })
    
    currency_format = workbook.add_format({
        'num_format': '$#,##0.00',
        'border': 1
    })
    
    revenue_format = workbook.add_format({
        'num_format': '$#,##0.0',
        'border': 1,
        'bold': True
    })
    
    metric_name_format = workbook.add_format({
        'border': 1,
        'align': 'left',
        'valign': 'vcenter'
    })
    
    section_header_format = workbook.add_format({
        'bold': True,
        'font_size': 12,
        'bg_color': '#44546A',
        'font_color': 'white',
        'border': 1,
        'align': 'left'
    })
    
    # Helper functions
    def get_excel_column(col_num):
        result = ""
        while col_num >= 0:
            result = chr(col_num % 26 + 65) + result
            col_num = col_num // 26 - 1
        return result
    
    def classify_metric_for_excel(metric_name):
        name_lower = metric_name.lower()
        is_rate = 'rate' in name_lower or metric_name.endswith(' %')
        
        if 'population' in name_lower and not any(kw in name_lower for kw in ['prevalent', 'prevalence', 'diagnosed', 'treated', 'treatment']):
            return 'total_population', is_rate
        elif 'prevalent' in name_lower or 'prevalence' in name_lower:
            return 'prevalence', is_rate
        elif 'diagnosed' in name_lower:
            return 'diagnosed', is_rate
        elif 'treated' in name_lower or 'treatment' in name_lower:
            return 'treated', is_rate
        else:
            return 'unknown', is_rate
    
    # ===== SHEET 1: FORECAST WITH EVENTS (MAIN SHEET WITH FORMULAS) =====
    events_sheet = workbook.add_worksheet('Forecast with Events')
    
    events_sheet.merge_range(0, 0, 0, len(epi_forecast.columns), 
                             'FORECAST WITH EVENTS APPLIED', title_format)
    events_sheet.set_row(0, 20)
    
    # Show applied events summary at top
    current_row = 2
    
    if market_events and len(market_events) > 0:
        events_sheet.merge_range(current_row, 0, current_row, len(epi_forecast.columns),
                                'APPLIED MARKET EVENTS', section_header_format)
        current_row += 1
        
        for event in market_events:
            impact_display = f"{event['impact_value']*100:+.1f}%" if event['impact_type'] == 'percentage_change' else f"{event['impact_value']:+,.0f}"
            event_text = f"• Year {event['event_year']}: {event['event_type']} - {event['target_metric']} ({impact_display})"
            events_sheet.write(current_row, 0, event_text, metric_name_format)
            current_row += 1
        
        current_row += 1
    
    if brand_events and len(brand_events) > 0:
        events_sheet.merge_range(current_row, 0, current_row, len(epi_forecast.columns),
                                'APPLIED BRAND EVENTS', section_header_format)
        current_row += 1
        
        for event in brand_events:
            event_text = f"• Year {event['event_year']}: {event['event_type']} - New Peak: {event['post_event_peak_share']*100:.1f}%, Curve: {event['curve_type']}"
            events_sheet.write(current_row, 0, event_text, metric_name_format)
            current_row += 1
        
        current_row += 1
    
    current_row += 1
    
    # ===== ADD PARAMETERS SECTION =====
    events_sheet.merge_range(current_row, 0, current_row, 1, 'MODEL PARAMETERS', header_format)
    current_row += 1
    
    # Base Population
    events_sheet.write(current_row, 0, 'Base Population:', metric_name_format)
    events_sheet.write(current_row, 1, base_pop, constant_format)
    base_pop_cell = f'B{current_row+1}'
    current_row += 1
    
    # Base Year
    events_sheet.write(current_row, 0, 'Base Year:', metric_name_format)
    events_sheet.write(current_row, 1, base_year, constant_format)
    current_row += 1
    
    # Population Growth Rate
    pop_growth_rate = growth_rates.get('Population', 0.007)
    events_sheet.write(current_row, 0, 'Population Growth (annual):', metric_name_format)
    events_sheet.write(current_row, 1, pop_growth_rate, constant_rate_format)
    pop_growth_cell = f'B{current_row+1}'
    current_row += 1
    
    # Base Rates
    events_sheet.write(current_row, 0, 'Prevalence Rate:', metric_name_format)
    events_sheet.write(current_row, 1, known_values.get('prevalence', 0.02), constant_rate_format)
    prev_rate_cell = f'B{current_row+1}'
    current_row += 1
    
    events_sheet.write(current_row, 0, 'Diagnosed Rate:', metric_name_format)
    events_sheet.write(current_row, 1, known_values.get('diagnosed', 0.85), constant_rate_format)
    diag_rate_cell = f'B{current_row+1}'
    current_row += 1
    
    events_sheet.write(current_row, 0, 'Treated Rate:', metric_name_format)
    events_sheet.write(current_row, 1, known_values.get('treated', 0.78), constant_rate_format)
    treat_rate_cell = f'B{current_row+1}'
    current_row += 1
    
    # Set column widths
    events_sheet.set_column('A:A', 35)
    events_sheet.set_column('B:B', 15)
    
    current_row += 1
    
    # ===== EPIDEMIOLOGY WITH EVENTS (WITH FORMULAS) =====
    events_sheet.merge_range(current_row, 0, current_row, len(epi_forecast.columns),
                            'EPIDEMIOLOGY FORECAST (WITH MARKET EVENTS)', section_header_format)
    current_row += 1
    
    # Headers
    events_sheet.write(current_row, 0, 'Metric', event_header_format)
    years = list(epi_forecast.columns)
    for col_idx, year in enumerate(years, start=1):
        events_sheet.write(current_row, col_idx, year, event_header_format)
        events_sheet.set_column(col_idx, col_idx, 15)
    current_row += 1
    
    # Track row positions for formula references
    epi_row_map = {}
    epi_data_start_row = current_row
    
    # Get metrics
    epi_metrics = [m for m in epi_forecast.index if m.strip() and not m.startswith('===')]
    
    # Epidemiology data WITH FORMULAS
    for metric in epi_metrics:
        events_sheet.write(current_row, 0, metric, metric_name_format)
        epi_row_map[metric] = current_row + 1  # Excel row (1-indexed)
        
        # Check if this is a derived/custom metric
        is_derived = metric in derived_metrics
        
        if is_derived:
            # Custom metric - handle specially
            config = derived_metrics[metric]
            source_metric = config.get('source')
            base_rate = config.get('base_rate', 0.0)
            is_rate_metric = config.get('is_rate', False)
            
            if is_rate_metric:
                # Custom rate - use actual value from forecast (which includes events)
                for col_idx in range(1, len(years) + 1):
                    # Get the actual value from the forecast dataframe
                    value = epi_forecast.loc[metric, years[col_idx-1]]
                    events_sheet.write_number(current_row, col_idx, float(value), constant_rate_format)
            else:
                # Custom population - formula based on source
                for col_idx in range(1, len(years) + 1):
                    col_letter = get_excel_column(col_idx)
                    source_row = epi_row_map.get(source_metric)
                    rate_metric_name = config.get('rate_metric')
                    
                    if not rate_metric_name:
                        # Try to construct rate metric name
                        if metric.endswith(' Count'):
                            base_name = metric[:-len(' Count')]
                            rate_metric_name = f"{base_name} %"
                        elif metric.endswith(' Population'):
                            base_name = metric[:-len(' Population')]
                            rate_metric_name = f"{base_name} Rate"
                    
                    rate_row = epi_row_map.get(rate_metric_name)
                    
                    if source_row and rate_row:
                        formula = f'={col_letter}{source_row}*{col_letter}{rate_row}'
                        events_sheet.write_formula(current_row, col_idx, formula, number_format)
                    elif source_row and base_rate > 0:
                        formula = f'={col_letter}{source_row}*{base_rate}'
                        events_sheet.write_formula(current_row, col_idx, formula, number_format)
                    else:
                        # Fallback to value
                        value = epi_forecast.loc[metric, years[col_idx-1]]
                        events_sheet.write_number(current_row, col_idx, float(value), number_format)
        else:
            # Standard metric
            metric_type, is_rate = classify_metric_for_excel(metric)
            
            # TOTAL POPULATION
            if metric_type == 'total_population' and not is_rate:
                for col_idx, year in enumerate(years, start=1):
                    year_val = int(year) if str(year).isdigit() else base_year
                    years_from_base = year_val - base_year
                    
                    if col_idx == 1:
                        formula = f'={base_pop_cell}'
                        events_sheet.write_formula(current_row, col_idx, formula, constant_format)
                    else:
                        col_letter = get_excel_column(1)
                        formula = f'={col_letter}{current_row + 1}*POWER(1+{pop_growth_cell},{years_from_base})'
                        events_sheet.write_formula(current_row, col_idx, formula, number_format)
            
            # PREVALENCE RATE
            elif metric_type == 'prevalence' and is_rate:
                for col_idx in range(1, len(years) + 1):
                    formula = f'={prev_rate_cell}'
                    events_sheet.write_formula(current_row, col_idx, formula, constant_rate_format)
            
            # PREVALENT POPULATION
            elif metric_type == 'prevalence' and not is_rate:
                for col_idx in range(1, len(years) + 1):
                    col_letter = get_excel_column(col_idx)
                    total_pop_row = None
                    prev_rate_row = None
                    
                    for m, r in epi_row_map.items():
                        m_type, m_is_rate = classify_metric_for_excel(m)
                        if m_type == 'total_population' and not m_is_rate:
                            total_pop_row = r
                        elif m_type == 'prevalence' and m_is_rate:
                            prev_rate_row = r
                    
                    if total_pop_row and prev_rate_row:
                        formula = f'={col_letter}{total_pop_row}*{col_letter}{prev_rate_row}'
                        events_sheet.write_formula(current_row, col_idx, formula, number_format)
                    else:
                        value = epi_forecast.loc[metric, years[col_idx-1]]
                        events_sheet.write_number(current_row, col_idx, float(value), number_format)
            
            # DIAGNOSED RATE
            elif metric_type == 'diagnosed' and is_rate:
                for col_idx in range(1, len(years) + 1):
                    formula = f'={diag_rate_cell}'
                    events_sheet.write_formula(current_row, col_idx, formula, constant_rate_format)
            
            # DIAGNOSED POPULATION
            elif metric_type == 'diagnosed' and not is_rate:
                for col_idx in range(1, len(years) + 1):
                    col_letter = get_excel_column(col_idx)
                    prev_pop_row = None
                    diag_rate_row = None
                    
                    for m, r in epi_row_map.items():
                        m_type, m_is_rate = classify_metric_for_excel(m)
                        if m_type == 'prevalence' and not m_is_rate:
                            prev_pop_row = r
                        elif m_type == 'diagnosed' and m_is_rate:
                            diag_rate_row = r
                    
                    if prev_pop_row and diag_rate_row:
                        formula = f'={col_letter}{prev_pop_row}*{col_letter}{diag_rate_row}'
                        events_sheet.write_formula(current_row, col_idx, formula, number_format)
                    else:
                        value = epi_forecast.loc[metric, years[col_idx-1]]
                        events_sheet.write_number(current_row, col_idx, float(value), number_format)
            
            # TREATED RATE
            elif metric_type == 'treated' and is_rate:
                for col_idx in range(1, len(years) + 1):
                    formula = f'={treat_rate_cell}'
                    events_sheet.write_formula(current_row, col_idx, formula, constant_rate_format)
            
            # TREATED POPULATION
            elif metric_type == 'treated' and not is_rate:
                for col_idx in range(1, len(years) + 1):
                    col_letter = get_excel_column(col_idx)
                    diag_pop_row = None
                    treat_rate_row = None
                    
                    for m, r in epi_row_map.items():
                        m_type, m_is_rate = classify_metric_for_excel(m)
                        if m_type == 'diagnosed' and not m_is_rate:
                            diag_pop_row = r
                        elif m_type == 'treated' and m_is_rate:
                            treat_rate_row = r
                    
                    if diag_pop_row and treat_rate_row:
                        formula = f'={col_letter}{diag_pop_row}*{col_letter}{treat_rate_row}'
                        events_sheet.write_formula(current_row, col_idx, formula, number_format)
                    else:
                        value = epi_forecast.loc[metric, years[col_idx-1]]
                        events_sheet.write_number(current_row, col_idx, float(value), number_format)
            
            # FALLBACK
            else:
                for col_idx, year in enumerate(years, start=1):
                    value = epi_forecast.loc[metric, year]
                    if is_rate:
                        events_sheet.write_number(current_row, col_idx, float(value), rate_format)
                    else:
                        events_sheet.write_number(current_row, col_idx, float(value), number_format)
        
        current_row += 1
    
    current_row += 2
    
    # ===== BRAND CONFIGURATION PARAMETERS =====
    events_sheet.merge_range(current_row, 0, current_row, 1, 'BRAND CONFIGURATION', section_header_format)
    current_row += 1
    
    events_sheet.write(current_row, 0, 'Launch Year:', config_label_format)
    events_sheet.write(current_row, 1, brand_config.get('launch_year', 2025), constant_format)
    launch_year_cell_brand = f'B{current_row + 1}'
    current_row += 1
    
    events_sheet.write(current_row, 0, 'Peak Share:', config_label_format)
    events_sheet.write(current_row, 1, brand_config.get('peak_share', 0.35), constant_rate_format)
    peak_share_cell_brand = f'B{current_row + 1}'
    current_row += 1
    
    events_sheet.write(current_row, 0, 'Uptake Type:', config_label_format)
    uptake_type_brand = brand_config.get('uptake_type', '7 Year Medium')
    events_sheet.write(current_row, 1, uptake_type_brand)
    current_row += 1
    
    events_sheet.write(current_row, 0, 'Adherence:', config_label_format)
    events_sheet.write(current_row, 1, brand_config.get('adherence', 0.68), constant_rate_format)
    adherence_cell_brand = f'B{current_row + 1}'
    current_row += 1
    
    events_sheet.write(current_row, 0, 'Annual Dose per Patient:', config_label_format)
    events_sheet.write(current_row, 1, brand_config.get('annual_dose', 360), constant_format)
    annual_dose_cell_brand = f'B{current_row + 1}'
    current_row += 1
    
    events_sheet.write(current_row, 0, 'Price per Dose ($):', config_label_format)
    events_sheet.write(current_row, 1, brand_config.get('price_per_dose', 16), currency_format)
    price_cell_brand = f'B{current_row + 1}'
    current_row += 1
    
    current_row += 1
    
    # ===== BRAND FORECAST WITH EVENTS (WITH FORMULAS) =====
    brand_name = brand_config.get('brand_name', 'Brand X')
    events_sheet.merge_range(current_row, 0, current_row, len(brand_forecast.columns),
                            f'{brand_name.upper()} FORECAST (WITH BRAND EVENTS)', section_header_format)
    current_row += 1
    
    # Headers
    events_sheet.write(current_row, 0, 'Metric', brand_header_format)
    for col_idx, year in enumerate(years, start=1):
        events_sheet.write(current_row, col_idx, year, brand_header_format)
    current_row += 1
    
    # Track brand rows for formulas
    brand_row_map = {}
    launch_year = brand_config.get('launch_year', 2025)
    target_population = brand_config.get('target_population', 'Treated patients')
    
    # Find target population row from epidemiology section
    target_pop_excel_row = None
    for metric, row_num in epi_row_map.items():
        if target_population.lower() in metric.lower():
            target_pop_excel_row = row_num
            break
    
    # Brand metrics with formulas
    brand_metrics_order = [
        'Brand X Patient Share',
        'Patients on Brand X',
        'Adherence',
        'Adherent Patients',
        'Annual dose per patient',
        'Price per dose ($)',
        'Total adherence adjusted Volume',
        'Revenue in billions'
    ]
    
    for metric in brand_metrics_order:
        if metric not in brand_forecast.index:
            continue
            
        events_sheet.write(current_row, 0, metric, metric_name_format)
        brand_row_map[metric] = current_row + 1  # Excel row (1-indexed)
        
        # ===== BRAND X PATIENT SHARE =====
        if metric == 'Brand X Patient Share':
            for col_idx, year in enumerate(years, start=1):
                # Use actual value from brand_forecast (already includes events)
                value = brand_forecast.loc[metric, year]
                events_sheet.write_number(current_row, col_idx, float(value), rate_format)
        
        # ===== PATIENTS ON BRAND X =====
        elif metric == 'Patients on Brand X':
            brand_share_row = brand_row_map.get('Brand X Patient Share')
            
            for col_idx in range(1, len(years) + 1):
                col_letter = get_excel_column(col_idx)
                
                if target_pop_excel_row and brand_share_row:
                    formula = f'={col_letter}${target_pop_excel_row}*{col_letter}${brand_share_row}'
                    events_sheet.write_formula(current_row, col_idx, formula, number_format)
                else:
                    value = brand_forecast.loc[metric, years[col_idx-1]]
                    events_sheet.write_number(current_row, col_idx, float(value), number_format)
        
        # ===== ADHERENCE =====
        elif metric == 'Adherence':
            for col_idx in range(1, len(years) + 1):
                formula = f'={adherence_cell_brand}'
                events_sheet.write_formula(current_row, col_idx, formula, constant_rate_format)
        
        # ===== ADHERENT PATIENTS =====
        elif metric == 'Adherent Patients':
            patients_row = brand_row_map.get('Patients on Brand X')
            adherence_row = brand_row_map.get('Adherence')
            
            for col_idx in range(1, len(years) + 1):
                col_letter = get_excel_column(col_idx)
                
                if patients_row and adherence_row:
                    formula = f'={col_letter}${patients_row}*{col_letter}${adherence_row}'
                    events_sheet.write_formula(current_row, col_idx, formula, number_format)
                else:
                    value = brand_forecast.loc[metric, years[col_idx-1]]
                    events_sheet.write_number(current_row, col_idx, float(value), number_format)
        
        # ===== ANNUAL DOSE PER PATIENT =====
        elif metric == 'Annual dose per patient':
            for col_idx in range(1, len(years) + 1):
                formula = f'={annual_dose_cell_brand}'
                events_sheet.write_formula(current_row, col_idx, formula, constant_format)
        
        # ===== PRICE PER DOSE =====
        elif metric == 'Price per dose ($)':
            for col_idx in range(1, len(years) + 1):
                formula = f'={price_cell_brand}'
                events_sheet.write_formula(current_row, col_idx, formula, currency_format)
        
        # ===== TOTAL ADHERENCE ADJUSTED VOLUME =====
        elif metric == 'Total adherence adjusted Volume':
            adherent_row = brand_row_map.get('Adherent Patients')
            dose_row = brand_row_map.get('Annual dose per patient')
            
            for col_idx in range(1, len(years) + 1):
                col_letter = get_excel_column(col_idx)
                
                if adherent_row and dose_row:
                    formula = f'={col_letter}${adherent_row}*{col_letter}${dose_row}'
                    events_sheet.write_formula(current_row, col_idx, formula, number_format)
                else:
                    value = brand_forecast.loc[metric, years[col_idx-1]]
                    events_sheet.write_number(current_row, col_idx, float(value), number_format)
        
        # ===== REVENUE IN BILLIONS =====
        elif metric == 'Revenue in billions':
            volume_row = brand_row_map.get('Total adherence adjusted Volume')
            price_row = brand_row_map.get('Price per dose ($)')
            
            for col_idx in range(1, len(years) + 1):
                col_letter = get_excel_column(col_idx)
                
                if volume_row and price_row:
                    formula = f'={col_letter}${volume_row}*{col_letter}${price_row}/1000000000'
                    events_sheet.write_formula(current_row, col_idx, formula, revenue_format)
                else:
                    value = brand_forecast.loc[metric, years[col_idx-1]]
                    events_sheet.write_number(current_row, col_idx, float(value), revenue_format)
        
        # ===== OTHER METRICS (if any exist) =====
        else:
            for col_idx, year in enumerate(years, start=1):
                value = brand_forecast.loc[metric, year]
                
                if 'volume' in metric.lower() or 'dose' in metric.lower():
                    events_sheet.write_number(current_row, col_idx, float(value), number_format)
                elif 'share' in metric.lower() or 'rate' in metric.lower() or 'adherence' in metric.lower():
                    events_sheet.write_number(current_row, col_idx, float(value), rate_format)
                elif 'revenue' in metric.lower() or 'billions' in metric.lower():
                    events_sheet.write_number(current_row, col_idx, float(value), revenue_format)
                elif 'price' in metric.lower() or '$' in metric:
                    events_sheet.write_number(current_row, col_idx, float(value), currency_format)
                else:
                    events_sheet.write_number(current_row, col_idx, float(value), number_format)
        
        current_row += 1    
    # ===== SHEET 2: FORECAST WITHOUT EVENTS (BASELINE COMPARISON) =====
    original_sheet = workbook.add_worksheet('Forecast without Events')
    
    original_sheet.merge_range(0, 0, 0, len(epi_original.columns), 
                               'ORIGINAL FORECAST (WITHOUT EVENTS) - BASELINE', title_format)
    original_sheet.set_row(0, 20)
    
    current_row = 2
    
    original_sheet.write(current_row, 0, 'This sheet shows your original forecast BEFORE any events were applied.', metric_name_format)
    original_sheet.write(current_row + 1, 0, 'Use this to compare against the "Forecast with Events" sheet.', metric_name_format)
    
    current_row += 3
    
    # ===== ORIGINAL EPIDEMIOLOGY =====
    original_sheet.merge_range(current_row, 0, current_row, len(epi_original.columns),
                              'EPIDEMIOLOGY FORECAST (ORIGINAL - NO MARKET EVENTS)', section_header_format)
    current_row += 1
    
    # Headers
    original_sheet.write(current_row, 0, 'Metric', original_header_format)
    for col_idx, year in enumerate(years, start=1):
        original_sheet.write(current_row, col_idx, year, original_header_format)
        original_sheet.set_column(col_idx, col_idx, 15)
    
    original_sheet.set_column('A:A', 35)
    current_row += 1
    
    # Original epidemiology data
    for metric in epi_original.index:
        if metric.strip() and not metric.startswith('==='):
            original_sheet.write(current_row, 0, metric, metric_name_format)
            
            for col_idx, year in enumerate(years, start=1):
                value = epi_original.loc[metric, year]
                
                if 'rate' in metric.lower() or '%' in metric:
                    original_sheet.write_number(current_row, col_idx, float(value), rate_format)
                else:
                    original_sheet.write_number(current_row, col_idx, float(value), number_format)
            
            current_row += 1
    
    current_row += 2
    
    # ===== ORIGINAL BRAND FORECAST =====
    original_sheet.merge_range(current_row, 0, current_row, len(brand_original.columns),
                              f'{brand_name.upper()} FORECAST (ORIGINAL - NO BRAND EVENTS)', section_header_format)
    current_row += 1
    
    # Headers
    original_sheet.write(current_row, 0, 'Metric', original_header_format)
    for col_idx, year in enumerate(years, start=1):
        original_sheet.write(current_row, col_idx, year, original_header_format)
    current_row += 1
    
    # Original brand data
    for metric in brand_original.index:
        original_sheet.write(current_row, 0, metric, metric_name_format)
        
        for col_idx, year in enumerate(years, start=1):
            value = brand_original.loc[metric, year]
            
            # Format based on metric type
            if 'volume' in metric.lower() or ('dose' in metric.lower() and 'price' not in metric.lower()):
                original_sheet.write_number(current_row, col_idx, float(value), number_format)
            elif 'share' in metric.lower() or 'rate' in metric.lower() or 'adherence' in metric.lower():
                original_sheet.write_number(current_row, col_idx, float(value), rate_format)
            elif 'revenue' in metric.lower() or 'billions' in metric.lower():
                original_sheet.write_number(current_row, col_idx, float(value), revenue_format)
            elif 'price' in metric.lower() or '$' in metric:
                original_sheet.write_number(current_row, col_idx, float(value), currency_format)
            else:
                original_sheet.write_number(current_row, col_idx, float(value), number_format)
        
        current_row += 1
    
    # ===== SHEET 3: SUMMARY =====
    summary_sheet = workbook.add_worksheet('Summary')
    summary_sheet.set_column('A:A', 25)
    summary_sheet.set_column('B:B', 30)
    
    row = 0
    summary_sheet.merge_range(f'A{row+1}:B{row+1}', 'FORECAST SUMMARY', title_format)
    row += 2
    
    summary_sheet.write(row, 0, 'Disease:', config_label_format)
    summary_sheet.write(row, 1, ai_assumptions.get('disease_name', 'N/A'))
    row += 1
    
    summary_sheet.write(row, 0, 'Geography:', config_label_format)
    summary_sheet.write(row, 1, ai_assumptions.get('geography', 'N/A'))
    row += 1
    
    summary_sheet.write(row, 0, 'Base Year:', config_label_format)
    summary_sheet.write(row, 1, ai_assumptions.get('base_year', 'N/A'))
    row += 2
    
    summary_sheet.merge_range(f'A{row+1}:B{row+1}', 'EVENTS APPLIED', section_header_format)
    row += 1
    
    if market_events and len(market_events) > 0:
        summary_sheet.write(row, 0, 'Market Events:', config_label_format)
        summary_sheet.write(row, 1, f"{len(market_events)} event(s) applied")
        row += 1
    else:
        summary_sheet.write(row, 0, 'Market Events:', config_label_format)
        summary_sheet.write(row, 1, "None")
        row += 1
    
    if brand_events and len(brand_events) > 0:
        summary_sheet.write(row, 0, 'Brand Events:', config_label_format)
        summary_sheet.write(row, 1, f"{len(brand_events)} event(s) applied")
        row += 1
    else:
        summary_sheet.write(row, 0, 'Brand Events:', config_label_format)
        summary_sheet.write(row, 1, "None")
        row += 1
    
    # ===== SHEET 4: MARKET EVENTS DETAIL =====
    if market_events and len(market_events) > 0:
        market_events_sheet = workbook.add_worksheet('Market Events Detail')
        
        market_events_sheet.merge_range(0, 0, 0, 5, 'MARKET EVENTS APPLIED', title_format)
        market_events_sheet.set_row(0, 20)
        
        # Headers
        headers = ['Year', 'Event Type', 'Description', 'Target Metric', 'Impact Type', 'Impact Value']
        for col_idx, header in enumerate(headers):
            market_events_sheet.write(2, col_idx, header, header_format)
        
        # Data
        for row_idx, event in enumerate(market_events, start=3):
            impact_display = f"{event['impact_value']*100:+.1f}%" if event['impact_type'] == 'percentage_change' else f"{event['impact_value']:+,.0f}"
            
            market_events_sheet.write(row_idx, 0, event['event_year'], number_format)
            market_events_sheet.write(row_idx, 1, event['event_type'])
            market_events_sheet.write(row_idx, 2, event.get('description', ''))
            market_events_sheet.write(row_idx, 3, event['target_metric'])
            market_events_sheet.write(row_idx, 4, 'Percentage Change' if event['impact_type'] == 'percentage_change' else 'Absolute Change')
            market_events_sheet.write(row_idx, 5, impact_display)
        
        # Set column widths
        market_events_sheet.set_column('A:A', 10)
        market_events_sheet.set_column('B:B', 25)
        market_events_sheet.set_column('C:C', 40)
        market_events_sheet.set_column('D:D', 30)
        market_events_sheet.set_column('E:E', 18)
        market_events_sheet.set_column('F:F', 15)
    
    # ===== SHEET 5: BRAND EVENTS DETAIL =====
    if brand_events and len(brand_events) > 0:
        brand_events_sheet = workbook.add_worksheet('Brand Events Detail')
        
        brand_events_sheet.merge_range(0, 0, 0, 6, 'BRAND EVENTS APPLIED', title_format)
        brand_events_sheet.set_row(0, 20)
        
        # Headers
        headers = ['Year', 'Event Type', 'Description', 'Post-Event Peak', 'Years to Peak', 'Curve Type', 'Curve Category']
        for col_idx, header in enumerate(headers):
            brand_events_sheet.write(2, col_idx, header, brand_header_format)
        
        # Data
        for row_idx, event in enumerate(brand_events, start=3):
            brand_events_sheet.write(row_idx, 0, event['event_year'], number_format)
            brand_events_sheet.write(row_idx, 1, event['event_type'])
            brand_events_sheet.write(row_idx, 2, event.get('description', ''))
            brand_events_sheet.write(row_idx, 3, event['post_event_peak_share'], rate_format)
            brand_events_sheet.write(row_idx, 4, event['years_to_peak'], number_format)
            brand_events_sheet.write(row_idx, 5, event['curve_type'])
            brand_events_sheet.write(row_idx, 6, event['curve_category'])
        
        # Set column widths
        brand_events_sheet.set_column('A:A', 10)
        brand_events_sheet.set_column('B:B', 25)
        brand_events_sheet.set_column('C:C', 40)
        brand_events_sheet.set_column('D:D', 18)
        brand_events_sheet.set_column('E:E', 15)
        brand_events_sheet.set_column('F:F', 25)
        brand_events_sheet.set_column('G:G', 20)
    
    # ===== SHEET 6: INSTRUCTIONS =====
    instructions_sheet = workbook.add_worksheet('Instructions')
    instructions_sheet.set_column('A:A', 80)
    
    instructions = [
        ['FORECAST WITH EVENTS - USER GUIDE', ''],
        ['', ''],
        ['OVERVIEW', ''],
        ['This workbook contains your complete forecast with all market and brand events applied.', ''],
        ['Epidemiology metrics use Excel formulas for full transparency.', ''],
        ['', ''],
        ['SHEETS EXPLAINED', ''],
        ['', ''],
        ['1. Forecast with Events', 'Main results showing epidemiology and brand forecast WITH all events applied (FORMULAS ACTIVE)'],
        ['2. Forecast without Events', 'BASELINE forecast BEFORE any events (for comparison)'],
        ['3. Summary', 'High-level overview of configuration and events'],
        ['4. Market Events Detail', 'Complete list of market events affecting epidemiology'],
        ['5. Brand Events Detail', 'Complete list of brand-specific events (LoE, competitors, etc.)'],
        ['6. Instructions', 'This sheet - how to use the model'],
        ['7. Research Citations', 'Data sources and research references'],
        ['', ''],
        ['EXCEL FORMULAS', ''],
        ['', ''],
        ['The "Forecast with Events" sheet uses Excel formulas for epidemiology calculations.', ''],
        ['This means you can:', ''],
        ['- Click any cell to see the formula in the formula bar', ''],
        ['- Trace precedents/dependents', ''],
        ['- Modify parameter values and see automatic updates', ''],
        ['- Perform sensitivity analysis', ''],
        ['', ''],
        ['HOW TO USE THIS MODEL', ''],
        ['', ''],
        ['COMPARING WITH AND WITHOUT EVENTS:', ''],
        ['- Open both "Forecast with Events" and "Forecast without Events" sheets side by side', ''],
        ['- Compare the same metric across both sheets to see event impact', ''],
        ['- The difference shows exactly how events changed your forecast', ''],
        ['', ''],
        ['SENSITIVITY ANALYSIS:', ''],
        ['- Modify parameter values in the MODEL PARAMETERS section', ''],
        ['- Watch the forecast update automatically via formulas', ''],
        ['', ''],
        ['UNDERSTANDING THE FORECAST', ''],
        ['', ''],
        ['The "Forecast with Events" sheet shows:', ''],
        ['- All market events that modified the epidemiology forecast', ''],
        ['- All brand events that affected market share dynamics', ''],
        ['- Updated epidemiology metrics after market events', ''],
        ['- Updated brand forecast after brand events', ''],
        ['', ''],
        ['The "Forecast without Events" sheet shows:', ''],
        ['- Your ORIGINAL baseline forecast', ''],
        ['- What the forecast would be WITHOUT any events', ''],
        ['- Use this as your comparison baseline', ''],
        ['', ''],
        ['EVENTS IMPACT', ''],
        ['', ''],
        ['Market Events:', ''],
        ['- Applied BEFORE brand calculations', ''],
        ['- Affect underlying epidemiology (e.g., treated population)', ''],
        ['- Examples: Regulatory changes, screening programs, treatment guidelines', ''],
        ['', ''],
        ['Brand Events:', ''],
        ['- Applied TO brand-specific metrics', ''],
        ['- Affect market share, uptake dynamics', ''],
        ['- Examples: Loss of Exclusivity, biosimilar entry, competitor launches', ''],
        ['', ''],
        ['QUICK COMPARISON TIPS', ''],
        ['', ''],
        ['To quickly assess event impact:', ''],
        ['1. Look at Revenue in final year - WITH vs WITHOUT events', ''],
        ['2. Compare Patient Volume trends', ''],
        ['3. Check Market Share evolution', ''],
        ['4. Review the "Market Events Detail" and "Brand Events Detail" sheets for context', ''],
        ['', ''],
        ['FORMULA APPROACH', ''],
        ['Impact = (Forecast WITH Events) - (Forecast WITHOUT Events)', ''],
        ['', ''],
        [f'Model generated: {datetime.now().strftime("%Y-%m-%d %H:%M:%S")}', '']
    ]
    
    for row_idx, (col1, col2) in enumerate(instructions):
        if any(keyword in col1 for keyword in ['GUIDE', 'OVERVIEW', 'SHEETS EXPLAINED', 'HOW TO USE', 'UNDERSTANDING', 'EVENTS IMPACT', 'QUICK COMPARISON', 'FORMULA APPROACH', 'EXCEL FORMULAS', 'SENSITIVITY ANALYSIS']):
            instructions_sheet.write(row_idx, 0, col1, title_format)
        else:
            instructions_sheet.write(row_idx, 0, col1)
        
        if col2:
            instructions_sheet.write(row_idx, 1, col2)
    
    # ===== SHEET 7: CITATIONS (if available) =====
    citations = ai_assumptions.get('citations', {})
    if citations:
        citations_sheet = workbook.add_worksheet('Research Citations')
        
        citations_sheet.merge_range(0, 0, 0, 2, 'RESEARCH CITATIONS & DATA SOURCES', title_format)
        citations_sheet.set_row(0, 20)
        
        citations_sheet.write(2, 0, 'Metric', header_format)
        citations_sheet.write(2, 1, 'Value', header_format)
        citations_sheet.write(2, 2, 'Source', header_format)
        
        citations_sheet.set_column('A:A', 25)
        citations_sheet.set_column('B:B', 15)
        citations_sheet.set_column('C:C', 80)
        
        row = 3
        known_values_cit = ai_assumptions.get('known_values', {})
        
        for metric, citation in citations.items():
            metric_display = metric.replace('_', ' ').title()
            citations_sheet.write(row, 0, metric_display, metric_name_format)
            
            value = known_values_cit.get(metric)
            if value:
                citations_sheet.write_number(row, 1, value, rate_format)
            else:
                citations_sheet.write(row, 1, 'N/A')
            
            citations_sheet.write(row, 2, citation, metric_name_format)
            row += 1
    
    writer.close()
    
    with open(filename, 'rb') as f:
        return f.read()
def load_model_from_excel(uploaded_file) -> Dict:
    """Load a previously exported model from Excel file - evaluates formulas
    
    Returns a dictionary with:
    - df_metrics: epidemiology forecast dataframe
    - df_brand: brand forecast dataframe (if exists)
    - brand_config: brand configuration dict
    - ai_assumptions: AI assumptions dict
    - input_data: parsed input data
    """
    
    try:
        import openpyxl
        from io import BytesIO
        import tempfile
        import os
        
        # Save uploaded file to a temporary file
        with tempfile.NamedTemporaryFile(delete=False, suffix='.xlsx') as tmp_file:
            tmp_file.write(uploaded_file.getvalue())
            tmp_path = tmp_file.name
        
        result = {
            'df_metrics': None,
            'df_brand': None,
            'brand_config': {},
            'ai_assumptions': {},
            'input_data': {},
            'derived_metrics': {}
        }
        
        excel_model = None  # Store reference to excel_model
        
        try:
            # ===== METHOD 1: Try using formulas library to evaluate =====
            if HAS_FORMULAS:
                st.info("Using formulas library to evaluate Excel formulas...")
                try:
                    # Load the Excel model
                    excel_model = ExcelModel()
                    excel_model.loads(tmp_path)
                    excel_model.finish()
                    
                    # This calculates all formulas
                    excel_model.calculate()
                    
                    st.success("✅ Formulas evaluated successfully!")
                    
                    # IMPORTANT: Load with data_only=True to get calculated values
                    # But first, we need to write the calculated values back
                    # OR use the excel_model to get values directly
                    
                except Exception as e:
                    st.warning(f"formulas library failed: {str(e)}, trying alternative method...")
                    excel_model = None
            
            # Load workbook - use data_only=True to get cached/calculated values
            # If formulas library worked, values should be in cache
            wb = openpyxl.load_workbook(tmp_path, data_only=True)
            
            # ===== LOAD SUMMARY AND CONFIGURATION =====
            available_sheets = wb.sheetnames
            st.write(f"Available sheets: {', '.join(available_sheets)}")
            
            if 'Summary' in available_sheets:
                ws = wb['Summary']
                
                # Read configuration
                for row in ws.iter_rows(min_row=1, max_row=100, values_only=True):
                    if not row[0]:
                        continue
                    
                    label = str(row[0])
                    value = row[1] if len(row) > 1 else None
                    
                    # Disease info
                    if label == 'Disease:':
                        result['ai_assumptions']['disease_name'] = value
                        result['input_data']['disease_name'] = value
                    elif label == 'Geography:':
                        result['ai_assumptions']['geography'] = value
                        result['input_data']['geography'] = value
                    elif label == 'Base Year:':
                        try:
                            result['ai_assumptions']['base_year'] = int(value)
                        except:
                            pass
                    
                    # Brand config
                    elif 'Brand Name' in label:
                        result['brand_config']['brand_name'] = str(value) if value else 'Brand X'
                    elif 'Launch Year' in label:
                        try:
                            result['brand_config']['launch_year'] = int(value)
                        except:
                            pass
                    elif 'Peak Market Share' in label or 'Peak Share' in label:
                        try:
                            if isinstance(value, str):
                                value = float(value.replace('%', '')) / 100
                            result['brand_config']['peak_share'] = float(value)
                        except:
                            pass
                    elif 'Uptake Curve' in label or 'Uptake Type' in label:
                        result['brand_config']['uptake_type'] = str(value) if value else '7 Year Medium'
                    elif 'Target Population' in label:
                        result['brand_config']['target_population'] = str(value) if value else 'Treated patients'
                    elif 'Adherence' in label and 'Rate' in label:
                        try:
                            if isinstance(value, str):
                                value = float(value.replace('%', '')) / 100
                            result['brand_config']['adherence'] = float(value)
                        except:
                            pass
                    elif 'Annual Dose' in label:
                        try:
                            result['brand_config']['annual_dose'] = int(value)
                        except:
                            pass
                    elif 'Price per Dose' in label:
                        try:
                            if isinstance(value, str):
                                value = float(value.replace('$', '').replace(',', ''))
                            result['brand_config']['price_per_dose'] = float(value)
                        except:
                            pass
            
            # ===== LOAD EPIDEMIOLOGY WITH FORMULA EVALUATION =====
            sheet_name = None
            if 'Epidemiology Forecast' in available_sheets:
                sheet_name = 'Epidemiology Forecast'
            elif 'Forecast Model' in available_sheets:
                sheet_name = 'Forecast Model'
            
            if sheet_name:
                ws = wb[sheet_name]
                
                # Read parameters from top rows
                for row_idx, row in enumerate(ws.iter_rows(min_row=1, max_row=15, values_only=True), start=1):
                    if len(row) < 2:
                        continue
                    
                    label = str(row[0]) if row[0] else ''
                    value = row[1]
                    
                    if 'Base Population' in label and value:
                        try:
                            result['ai_assumptions']['population'] = int(value)
                            result['input_data']['population'] = int(value)
                        except:
                            pass
                    elif 'Population Growth' in label and value:
                        try:
                            if 'growth_rates' not in result['ai_assumptions']:
                                result['ai_assumptions']['growth_rates'] = {}
                            result['ai_assumptions']['growth_rates']['Population'] = float(value)
                        except:
                            pass
                    elif 'Prevalence Rate' in label and value:
                        try:
                            if 'known_values' not in result['ai_assumptions']:
                                result['ai_assumptions']['known_values'] = {}
                            result['ai_assumptions']['known_values']['prevalence'] = float(value)
                        except:
                            pass
                    elif 'Diagnosed Rate' in label and value:
                        try:
                            if 'known_values' not in result['ai_assumptions']:
                                result['ai_assumptions']['known_values'] = {}
                            result['ai_assumptions']['known_values']['diagnosed'] = float(value)
                        except:
                            pass
                    elif 'Treated Rate' in label and value:
                        try:
                            if 'known_values' not in result['ai_assumptions']:
                                result['ai_assumptions']['known_values'] = {}
                            result['ai_assumptions']['known_values']['treated'] = float(value)
                        except:
                            pass
                
                # Find data section
                metric_header_row = None
                for row_idx, row in enumerate(ws.iter_rows(min_row=1, values_only=True), start=1):
                    if row[0] == 'Metric':
                        metric_header_row = row_idx
                        break
                
                if metric_header_row:
                    st.info(f"Found data at row {metric_header_row}")
                    
                    # Get year columns from header row  
                    header_row = list(ws[metric_header_row])
                    year_columns = {}
                    for col_idx, cell in enumerate(header_row[1:], start=2):  # Skip first column (Metric)
                        if cell.value and str(cell.value).isdigit():
                            year_columns[col_idx] = str(cell.value)
                    
                    st.write(f"Years found: {list(year_columns.values())}")
                    
                    # Read metrics data - NOW WITH PROPER FORMULA EVALUATION
                    metrics_data = {}
                    
                    # OPTION 1: If formulas library worked, use it to get calculated values
                    if excel_model is not None:
                        st.info("Reading calculated values from formulas library...")
                        
                        for row_idx, row in enumerate(ws.iter_rows(min_row=metric_header_row + 1), start=metric_header_row + 1):
                            metric_name = row[0].value
                            
                            if not metric_name or str(metric_name).strip() == '' or str(metric_name).startswith('==='):
                                continue
                            
                            # Stop if we hit brand section
                            if 'BRAND' in str(metric_name).upper() or 'FORECAST' in str(metric_name).upper():
                                break
                            
                            metric_values = {}
                            
                            for col_idx, year in year_columns.items():
                                # Get cell address in Excel notation (e.g., "B10")
                                col_letter = openpyxl.utils.get_column_letter(col_idx)
                                cell_address = f"{sheet_name}!{col_letter}{row_idx}"
                                
                                try:
                                    # Get calculated value from excel_model
                                    calculated_value = excel_model[cell_address].value
                                    
                                    # Handle different value types
                                    if calculated_value is None:
                                        metric_values[year] = 0.0
                                    elif isinstance(calculated_value, (int, float)):
                                        metric_values[year] = float(calculated_value)
                                    else:
                                        # Try to convert to float
                                        try:
                                            metric_values[year] = float(calculated_value)
                                        except:
                                            metric_values[year] = 0.0
                                except Exception as e:
                                    # Fallback to cell value
                                    cell = row[col_idx - 1]
                                    if cell.value is not None and not (isinstance(cell.value, str) and cell.value.startswith('=')):
                                        try:
                                            metric_values[year] = float(cell.value)
                                        except:
                                            metric_values[year] = 0.0
                                    else:
                                        metric_values[year] = 0.0
                            
                            if metric_values:
                                metrics_data[str(metric_name)] = metric_values
                    
                    else:
                        # OPTION 2: Formulas library not available - use data_only mode (cached values)
                        st.info("Reading cached values from Excel (data_only mode)...")
                        
                        for row in ws.iter_rows(min_row=metric_header_row + 1, values_only=True):
                            metric_name = row[0]
                            
                            if not metric_name or str(metric_name).strip() == '' or str(metric_name).startswith('==='):
                                continue
                            
                            # Stop if we hit brand section
                            if 'BRAND' in str(metric_name).upper() or 'FORECAST' in str(metric_name).upper():
                                break
                            
                            metric_values = {}
                            
                            for col_idx, year in year_columns.items():
                                value = row[col_idx - 1]  # Adjust for 0-indexing
                                
                                if value is not None:
                                    try:
                                        metric_values[year] = float(value)
                                    except:
                                        metric_values[year] = 0.0
                                else:
                                    metric_values[year] = 0.0
                            
                            if metric_values:
                                metrics_data[str(metric_name)] = metric_values
                    
                    if metrics_data:
                        result['df_metrics'] = pd.DataFrame(metrics_data).T
                        st.success(f"✅ Loaded {len(result['df_metrics'])} epidemiology metrics")
                        
                        # Check if we got real values
                        sample_value = result['df_metrics'].iloc[0, 0] if len(result['df_metrics']) > 0 else 0
                        st.write(f"Sample value (should not be 0): {sample_value}")
                        
                        # Show first few values for debugging
                        st.write("First row of data:")
                        st.write(result['df_metrics'].iloc[0])
            
            # ===== LOAD BRAND FORECAST =====
            if 'Brand Forecast' in available_sheets:
                ws = wb['Brand Forecast']
                
                # Find metric header
                metric_header_row = None
                for row_idx, row in enumerate(ws.iter_rows(min_row=1, values_only=True), start=1):
                    if row[0] == 'Metric':
                        metric_header_row = row_idx
                        break
                
                if metric_header_row:
                    header_row = list(ws[metric_header_row])
                    year_columns = {}
                    for col_idx, cell in enumerate(header_row[1:], start=2):
                        if cell.value and str(cell.value).isdigit():
                            year_columns[col_idx] = str(cell.value)
                    
                    brand_data = {}
                    
                    # Use similar approach as epidemiology data
                    if excel_model is not None:
                        for row_idx, row in enumerate(ws.iter_rows(min_row=metric_header_row + 1), start=metric_header_row + 1):
                            metric_name = row[0].value
                            
                            if not metric_name or str(metric_name).strip() == '':
                                continue
                            
                            metric_values = {}
                            
                            for col_idx, year in year_columns.items():
                                col_letter = openpyxl.utils.get_column_letter(col_idx)
                                cell_address = f"Brand Forecast!{col_letter}{row_idx}"
                                
                                try:
                                    calculated_value = excel_model[cell_address].value
                                    if calculated_value is not None:
                                        metric_values[year] = float(calculated_value)
                                    else:
                                        metric_values[year] = 0.0
                                except:
                                    cell = row[col_idx - 1]
                                    if cell.value is not None and not (isinstance(cell.value, str) and cell.value.startswith('=')):
                                        try:
                                            metric_values[year] = float(cell.value)
                                        except:
                                            metric_values[year] = 0.0
                                    else:
                                        metric_values[year] = 0.0
                            
                            if metric_values:
                                brand_data[str(metric_name)] = metric_values
                    else:
                        for row in ws.iter_rows(min_row=metric_header_row + 1, values_only=True):
                            metric_name = row[0]
                            
                            if not metric_name or str(metric_name).strip() == '':
                                continue
                            
                            metric_values = {}
                            
                            for col_idx, year in year_columns.items():
                                value = row[col_idx - 1]
                                
                                if value is not None:
                                    try:
                                        metric_values[year] = float(value)
                                    except:
                                        metric_values[year] = 0.0
                                else:
                                    metric_values[year] = 0.0
                            
                            if metric_values:
                                brand_data[str(metric_name)] = metric_values
                    
                    if brand_data:
                        result['df_brand'] = pd.DataFrame(brand_data).T
                        st.success(f"✅ Loaded {len(result['df_brand'])} brand metrics")
            
            # ===== IDENTIFY CUSTOM METRICS =====
            if result['df_metrics'] is not None:
                derived_metrics = {}
                metrics_list = list(result['df_metrics'].index)
                
                for metric in metrics_list:
                    metric_str = str(metric)
                    
                    if metric_str.endswith(' %'):
                        base_name = metric_str[:-2]
                        metric_idx = metrics_list.index(metric)
                        
                        if metric_idx > 0:
                            source_metric = None
                            for i in range(metric_idx - 1, -1, -1):
                                potential_source = metrics_list[i]
                                if not (potential_source.endswith(' %') or potential_source.endswith(' Count') or potential_source.endswith(' Population')):
                                    source_metric = potential_source
                                    break
                            
                            if source_metric:
                                rate_values = result['df_metrics'].loc[metric].values
                                def safe_rate_convert(vals):
                                    if not vals or len(vals) == 0:
                                        return 0.0
                                    try:
                                        val = float(vals[0]) if vals[0] is not None else 0.0
                                        return val if val != 0 else 0.0
                                    except (ValueError, TypeError):
                                        return 0.0
                                base_rate = safe_rate_convert(rate_values)
                                
                                derived_metrics[metric_str] = {
                                    'source': source_metric,
                                    'base_rate': base_rate,
                                    'growth_rate': 0.0,
                                    'is_rate': True
                                }
                                
                                count_metric = f"{base_name} Count"
                                pop_metric = f"{base_name} Population"
                                
                                if count_metric in metrics_list:
                                    derived_metrics[count_metric] = {
                                        'source': source_metric,
                                        'rate_metric': metric_str,
                                        'base_rate': base_rate,
                                        'growth_rate': 0.0,
                                        'is_rate': False
                                    }
                                elif pop_metric in metrics_list:
                                    derived_metrics[pop_metric] = {
                                        'source': source_metric,
                                        'rate_metric': metric_str,
                                        'base_rate': base_rate,
                                        'growth_rate': 0.0,
                                        'is_rate': False
                                    }
                
                if derived_metrics:
                    result['derived_metrics'] = derived_metrics
                    result['ai_assumptions']['derived_metrics'] = derived_metrics
            
            # Build metrics list
            if result['df_metrics'] is not None:
                result['ai_assumptions']['metrics'] = list(result['df_metrics'].index)
            
        finally:
            # Clean up temp file
            wb.close() if 'wb' in locals() else None
            try:
                os.unlink(tmp_path)
            except:
                pass
        
        return result
        
    except Exception as e:
        st.error(f"Error loading model: {str(e)}")
        import traceback
        st.code(traceback.format_exc())
        return None

    

def main():
    # Apply custom CSS
    apply_custom_css()
    
    # Initialize conversation state
    init_conversation_state()
    
    # Company logo at the top
    try:
        st.image("Axtria.png", width=200)
    except:
        pass  # If logo file doesn't exist, continue without it
    
    # Header banner with info button
    col_header1, col_header2 = st.columns([6, 1])
    with col_header1:
        st.markdown("""
        <div class="app-header">
            <h1>Epidemiology Forecast Platform</h1>
            <p>AI-powered epidemiology and brand forecasting system</p>
        </div>
        """, unsafe_allow_html=True)
    with col_header2:
        with st.popover("ℹ️ Instructions", use_container_width=True):
            st.markdown("""
            **How to use:**
            
            1. **Natural Language Input**: Describe your forecast in plain text
               - Minimum: Disease name and geography
               - Optional: Add "with funnel" for auto-discovery of segments
               - Optional: Add "with brand analysis" for targeting insights
            
            2. **System Auto-Discovery**:
               - Discovers missing epidemiology rates from medical literature
               - Suggests clinically relevant patient segments when requested
               - Recommends optimal brand targeting parameters
            
            3. **Review & Export**:
               - Review generated forecasts in the tabs below
               - Export to Excel with full calculations
               - Save scenarios for comparison
            
            **Input Format Examples**:
            - Minimal: "Diabetes United States 2025-2035"
            - With segments: "Obesity USA 2025-2035 with funnel"
            - With brand: "Heart failure Europe 2025-2035 with brand analysis"
            - Complete: "COPD USA 2025-2035 with funnel and brand analysis"
            """)
    
    # ===== NEW: AGENTIC INTERFACE =====
    if st.session_state.get('agentic_mode', True):
        st.markdown('<div class="section-header"><h2>Forecast Input</h2></div>', unsafe_allow_html=True)
        
        # Show conversation history if available
        if st.session_state.get('conversation_history'):
            with st.expander("Conversation History", expanded=False):
                for turn in st.session_state['conversation_history'][-3:]:  # Show last 3 turns
                    st.markdown(f"**{turn['timestamp']} - You:** {turn['user_input']}")
                    st.markdown(f"**Assistant:** {turn['assistant_response']}")
                    st.markdown("---")
        
        # Check if there are pending clarifications
        if st.session_state.get('pending_clarifications'):
            st.markdown("### Clarification Required")
            questions = st.session_state['pending_clarifications']
            
            # Validate questions format to prevent errors
            if not isinstance(questions, list):
                st.error(f"Invalid questions format: {type(questions)} - {str(questions)[:200]}...")
                st.session_state['pending_clarifications'] = []
                return
            
            valid_questions = []
            for i, q in enumerate(questions):
                if not isinstance(q, dict):
                    st.error(f"Invalid question format at index {i}: {type(q)} - {str(q)}")
                    continue
                if 'question' not in q:
                    st.error(f"Missing 'question' field at index {i}: {q}")
                    continue
                valid_questions.append(q)
            
            if not valid_questions:
                st.error("No valid questions found. Please try your request again.")
                st.session_state['pending_clarifications'] = []
                return
            
            st.markdown("Please provide the following information to complete your forecast:")
            
            # Create form for clarifications
            with st.form("clarification_form"):
                responses = {}
                for i, q in enumerate(valid_questions):
                    st.markdown(f"**{i+1}. {q['question']}**")
                    if q.get('explanation'):
                        st.caption(q['explanation'])
                    
                    if q.get('suggested_default'):
                        responses[q['parameter']] = st.text_input(
                            f"Answer {i+1}:",
                            value=q['suggested_default'],
                            key=f"clarify_{i}"
                        )
                    else:
                        responses[q['parameter']] = st.text_input(
                            f"Answer {i+1}:",
                            key=f"clarify_{i}"
                        )
                
                if st.form_submit_button("Submit Answers"):
                    # Update the parsed data with clarifications
                    if 'extracted_parameters' in st.session_state.get('conversation_context', {}):
                        for param, value in responses.items():
                            if param == 'population' and value.isdigit():
                                st.session_state['conversation_context']['extracted_parameters']['population'] = int(value)
                            # Add more parameter handling as needed
                    
                    # Clear pending clarifications
                    st.session_state['pending_clarifications'] = []
                    
                    # Retry forecast generation with clarified data
                    clarified_prompt = f"{st.session_state['conversation_context']['original_prompt']} [Clarified: {responses}]"
                    result = orchestrate_agentic_forecast(clarified_prompt)
                    
                    if result['status'] == 'success':
                        st.success("🎉 Forecast generated successfully with your clarifications!")
                        st.rerun()
        
        # Main agentic input section
        elif not st.session_state.get('conversation_context', {}).get('forecast_complete'):
            # Natural language input
            user_prompt = st.text_area(
                "Describe your forecast requirements:",
                height=120,
                placeholder="Example: Diabetes United States 2025-2035 with funnel",
                key="agentic_prompt"
            )
            
            col1, col2, col3 = st.columns([1, 1, 2])
            with col1:
                if st.button("Generate Forecast", type="primary", disabled=not user_prompt.strip()):
                    with st.status("Processing your request...", expanded=True):
                        result = orchestrate_agentic_forecast(user_prompt)
                        
                        if result['status'] == 'needs_clarification':
                            st.info(f"Need clarification on {len(result['questions'])} parameters.")
                        elif result['status'] == 'success':
                            st.success("Forecast generated successfully!")
                        elif result['status'] == 'error':
                            st.error(f"Error: {result['message']}")
                    st.rerun()
            
            with col2:
                if st.button("Switch to Manual Mode"):
                    st.session_state['agentic_mode'] = False
                    st.rerun()
        
        # Show results if forecast is complete
        elif st.session_state.get('conversation_context', {}).get('forecast_complete'):
            st.markdown("### Forecast Complete")
            
            # Show quick summary
            forecast_state = st.session_state.get('current_forecast_state', {})
            col1, col2, col3 = st.columns(3)
            with col1:
                if forecast_state.get('epidemiology_complete'):
                    st.success("Epidemiology Complete")
                else:
                    st.warning("Epidemiology Incomplete")
            
            with col2:
                if forecast_state.get('brand_complete'):
                    st.success("Brand Forecast Complete")
                else:
                    st.info("Brand Forecast Not Available")
                    
            with col3:
                if forecast_state.get('has_custom_segments'):
                    st.success("Custom Segments Added")
                else:
                    st.info("No Custom Segments")
            
            # Conversation input for adjustments
            st.markdown("### Make Adjustments")
            adjustment_prompt = st.text_input(
                "Modify parameters:",
                placeholder="Example: increase adherence to 75%, add biosimilar in 2029",
                key="adjustment_prompt"
            )
            
            col1, col2, col3, col4 = st.columns([1, 1, 1, 2])
            with col1:
                if st.button("Update", disabled=not adjustment_prompt.strip()):
                    # Detect parameter changes
                    changes = detect_parameter_changes(adjustment_prompt)
                    if changes.get('has_changes'):
                        with st.spinner("Applying changes and regenerating forecast..."):
                            # Apply changes to current data
                            success = apply_conversational_changes(changes)
                            if success:
                                add_conversation_turn(
                                    adjustment_prompt,
                                    "Changes applied successfully! Forecast updated.",
                                    'adjustment'
                                )
                                st.success("Changes applied successfully!")
                                st.rerun()
                            else:
                                st.error("Failed to apply changes. Please try rephrasing your request.")
                    else:
                        st.info("No parameter changes detected. Please be more specific.")
            
            with col2:
                if st.button("New Forecast"):
                    # Reset conversation state
                    for key in ['conversation_context', 'current_forecast_state', 'pending_clarifications']:
                        if key in st.session_state:
                            del st.session_state[key]
                    st.rerun()
                    
            with col3:
                if st.button("View Results", type="secondary"):
                    # Jump to results section
                    st.session_state['show_results_section'] = True
    
    # ===== MANUAL MODE TOGGLE =====
    else:
        st.markdown('<div class="section-header"><h2>Manual Configuration Mode</h2></div>', unsafe_allow_html=True)
        
        col1, col2 = st.columns([1, 3])
        with col1:
            if st.button("Switch to AI Mode"):
                st.session_state['agentic_mode'] = True
                st.rerun()
        
        with col2:
            st.info("Switch to AI Mode for natural language forecasting, or continue with manual configuration below.")
    
    # Show results section if forecast is complete or manual mode
    if (st.session_state.get('conversation_context', {}).get('forecast_complete') or 
        not st.session_state.get('agentic_mode', True) or 
        st.session_state.get('show_results_section')):
        
        # Continue with existing manual interface
        pass  # Placeholder - existing UI continues below
    
    # ===== EXISTING UPLOAD MODEL SECTION =====
    st.markdown('<div class="section-header"><h2>Load Existing Model</h2></div>', unsafe_allow_html=True)
    
    with st.expander("Upload Previously Downloaded Model", expanded=False):
        st.markdown("""
        <div class="info-card">
            <strong>Upload Existing Forecast</strong><br>
            Have you previously downloaded a complete model from this application? 
            Upload it here to continue working with it!
            <br><br>
            <strong>Supported files:</strong>
            <ul>
                <li>Excel files (.xlsx) exported using "Download Complete Excel Model" button</li>
                <li>The uploaded model will restore your epidemiology forecast, brand forecast, and all configurations</li>
            </ul>
        </div>
        """, unsafe_allow_html=True)
        
        uploaded_file = st.file_uploader(
            "Choose Excel file",
            type=['xlsx'],
            key='model_upload',
            help="Upload an Excel file that was previously downloaded from this application"
        )
        
        if uploaded_file is not None:
            with st.spinner('Loading model...'):
                loaded_data = load_model_from_excel(uploaded_file)
                
                if loaded_data:
                    st.success("✅ Model loaded successfully!")
                    
                    # Display what was loaded
                    col1, col2, col3 = st.columns(3)
                    
                    with col1:
                        if loaded_data['df_metrics'] is not None:
                            st.metric("📊 Epidemiology Metrics", len(loaded_data['df_metrics']))
                        else:
                            st.warning("⚠️ No epidemiology data found")
                    
                    with col2:
                        if loaded_data['df_brand'] is not None:
                            st.metric("💊 Brand Metrics", len(loaded_data['df_brand']))
                        else:
                            st.info("ℹ️ No brand forecast found")
                    
                    with col3:
                        if loaded_data['derived_metrics']:
                            st.metric("🎯 Custom Metrics", len(loaded_data['derived_metrics']))
                        else:
                            st.info("ℹ️ No custom metrics found")
                    
                    # Show configuration summary
                    st.markdown("---")
                    st.markdown("**📋 Loaded Configuration:**")
                    
                    config_col1, config_col2 = st.columns(2)
                    
                    with config_col1:
                        st.markdown("**Disease Information:**")
                        st.write(f"- Disease: {loaded_data['ai_assumptions'].get('disease_name', 'N/A')}")
                        st.write(f"- Geography: {loaded_data['ai_assumptions'].get('geography', 'N/A')}")
                        st.write(f"- Base Year: {loaded_data['ai_assumptions'].get('base_year', 'N/A')}")
                        st.write(f"- Population: {loaded_data['ai_assumptions'].get('population', 0):,}")
                    
                    with config_col2:
                        if loaded_data['brand_config']:
                            st.markdown("**Brand Configuration:**")
                            st.write(f"- Brand: {loaded_data['brand_config'].get('brand_name', 'N/A')}")
                            st.write(f"- Launch Year: {loaded_data['brand_config'].get('launch_year', 'N/A')}")
                            st.write(f"- Peak Share: {loaded_data['brand_config'].get('peak_share', 0)*100:.1f}%")
                            st.write(f"- Price per Dose: ${loaded_data['brand_config'].get('price_per_dose', 0):.2f}")
                    
                    st.markdown("---")
                    
                    # Button to load into session state
                    # Button to load into session state
                    # Button to load into session state
                    if st.button("✅ Load This Model into Application", type="primary", use_container_width=True):
                        # Show what's being loaded
                        st.write("**Debug: Loading data...**")
                        
                        # Load epidemiology data
                        if loaded_data['df_metrics'] is not None:
                            st.write(f"Debug: df_metrics shape: {loaded_data['df_metrics'].shape}")
                            st.write(f"Debug: df_metrics sample:")
                            st.write(loaded_data['df_metrics'].head())
                            st.write(f"Debug: First value: {loaded_data['df_metrics'].iloc[0, 0]}")
                            
                            st.session_state['df_metrics_base'] = loaded_data['df_metrics'].copy()
                            st.session_state['df_metrics_custom'] = loaded_data['df_metrics'].copy()
                        
                        # Load brand data
                        if loaded_data['df_brand'] is not None:
                            st.write(f"Debug: df_brand shape: {loaded_data['df_brand'].shape}")
                            st.write(f"Debug: df_brand sample:")
                            st.write(loaded_data['df_brand'].head())
                            
                            st.session_state['df_brand_forecast'] = loaded_data['df_brand'].copy()
                        
                        # Load configurations
                        if loaded_data['brand_config']:
                            st.write(f"Debug: brand_config: {loaded_data['brand_config']}")
                            st.session_state['brand_config'] = loaded_data['brand_config'].copy()
                            
                            # Pre-populate brand input fields
                            st.session_state['brand_name'] = loaded_data['brand_config'].get('brand_name', 'Brand X')
                            st.session_state['launch_year'] = loaded_data['brand_config'].get('launch_year', 2026)
                            st.session_state['peak_share'] = loaded_data['brand_config'].get('peak_share', 0.35) * 100
                            st.session_state['uptake_type'] = loaded_data['brand_config'].get('uptake_type', '7 Year Medium')
                            st.session_state['target_population'] = loaded_data['brand_config'].get('target_population', 'Treated patients')
                            st.session_state['adherence'] = loaded_data['brand_config'].get('adherence', 0.68) * 100
                            st.session_state['annual_dose'] = loaded_data['brand_config'].get('annual_dose', 360)
                            st.session_state['price_per_dose'] = loaded_data['brand_config'].get('price_per_dose', 16.0)
                        
                        # Load AI assumptions
                        if loaded_data['ai_assumptions']:
                            st.write(f"Debug: ai_assumptions keys: {loaded_data['ai_assumptions'].keys()}")
                            st.session_state['ai_assumptions_v2'] = loaded_data['ai_assumptions'].copy()
                        
                        # Load input data and reconstruct input text
                        if loaded_data['input_data']:
                            st.session_state['parsed'] = loaded_data['input_data'].copy()
                            
                            # Reconstruct input text with ALL available information
                            input_lines = []
                            if loaded_data['input_data'].get('disease_name'):
                                input_lines.append(f"Disease: {loaded_data['input_data']['disease_name']}")
                            if loaded_data['input_data'].get('geography'):
                                input_lines.append(f"Geography: {loaded_data['input_data']['geography']}")
                            if loaded_data['input_data'].get('population'):
                                input_lines.append(f"Population: {loaded_data['input_data']['population']}")
                            
                            # Add known values if available
                            known_values = loaded_data['ai_assumptions'].get('known_values', {})
                            if known_values.get('prevalence'):
                                input_lines.append(f"Prevalence: {known_values['prevalence']*100:.2f}%")
                            if known_values.get('diagnosed'):
                                input_lines.append(f"Diagnosed: {known_values['diagnosed']*100:.2f}%")
                            if known_values.get('treated'):
                                input_lines.append(f"Treated: {known_values['treated']*100:.2f}%")
                            
                            if input_lines:
                                st.session_state['input_text'] = '\n'.join(input_lines)
                        
                        # Set years from the dataframe
                        if loaded_data['df_metrics'] is not None:
                            years = [int(col) for col in loaded_data['df_metrics'].columns if str(col).isdigit()]
                            if years:
                                st.session_state['start_year'] = min(years)
                                st.session_state['end_year'] = max(years)
                        
                        # Set frequency (default to yearly)
                        if 'frequency' not in st.session_state:
                            st.session_state['frequency'] = 'Yearly'
                        
                        # Mark progress steps as completed
                        st.session_state['step'] = 'forecast_generated'
                        st.session_state['model_loaded_from_file'] = True
                        
                        st.success("🎉 Model loaded into application! You can now view and modify it below.")
                        st.info("✏️ All input fields, parameters, and forecasts have been populated. Scroll down to view and edit them!")
                        
                        # Don't rerun yet - let user see the debug info
                        st.write("**Check the data above. If it looks correct, the page will reload in a moment.**")
                        import time
                        time.sleep(4)
                        st.rerun()    
    st.markdown("---")
    
    # Sidebar navigation and progress
    with st.sidebar:
        st.markdown("### Navigation")
        show_progress_indicator()
        
        st.markdown("---")
        st.markdown("### ⚙️ Settings")
        
        # Dark mode toggle
        if 'dark_mode' not in st.session_state:
            st.session_state['dark_mode'] = False
        
        # Help section
        with st.expander("❓ Help & Support", expanded=False):
            st.markdown("""
            **Need Help?**
            
            - 📖 [View Documentation](#)
            - 💬 [Contact Support](#)
            - 🐛 [Report Issue](#)
            
            **Keyboard Shortcuts:**
            - `Ctrl/Cmd + Enter` - Submit form
            - `Ctrl/Cmd + S` - Save scenario
            """)
    
    # Show welcome screen for first-time users
    show_welcome_screen()
    
    # Main input section with improved styling
    st.markdown('<div class="section-header"><span class="section-icon">📝</span><h2>Step 1: Input Disease Information</h2></div>', unsafe_allow_html=True)
    st.markdown(f"""
    <div class="info-card">
        <strong>📋 Input Format:</strong> Enter your data as key-value pairs (one per line).
        <br><br>
        <strong>Example:</strong><br>
        <code>{EXAMPLE_HIV.replace(chr(10), '<br>')}</code>
    </div>
    """, unsafe_allow_html=True)

    # Step 0: parse input
    input_text = st.text_area(
        'Enter Disease Data',
        height=180,
        key='input_text',
        placeholder=EXAMPLE_HIV,
        help="Enter one parameter per line in 'Key: Value' format"
    )
    
    # Quick fill examples
    # col_ex1, col_ex2, col_ex3 = st.columns(3)
    # with col_ex1:
    #     if st.button("📋 Example: HIV", help="Load HIV example data"):
    #         st.session_state['input_text'] = EXAMPLE_HIV
    #         st.rerun()
    # with col_ex2:
    #     if st.button("📋 Example: Diabetes", help="Load Diabetes example data"):
    #         st.session_state['input_text'] = EXAMPLE_DIABETES
    #         st.rerun()
    # with col_ex3:
    #     if st.button("📋 Example: Cardiovascular", help="Load CVD example data"):
    #         st.session_state['input_text'] = EXAMPLE_CVD
    #         st.rerun()

    st.markdown("---")
    st.markdown('<div class="section-header"><span class="section-icon">⚙️</span><h3>Forecast Configuration</h3></div>', unsafe_allow_html=True)
    
    col1, col2, col3 = st.columns(3)
    with col1:
        # Get current frequency from session state, handle case mismatch
        current_frequency = st.session_state.get('frequency', 'Yearly')
        # Normalize to match selectbox options
        if current_frequency.lower() == 'yearly':
            current_frequency = 'Yearly'
        elif current_frequency.lower() == 'quarterly':
            current_frequency = 'Quarterly'  
        elif current_frequency.lower() == 'monthly':
            current_frequency = 'Monthly'
        
        frequency = st.selectbox(
            '📅 Frequency',
            ['Yearly','Quarterly','Monthly'],
            index=['Yearly','Quarterly','Monthly'].index(current_frequency) if current_frequency in ['Yearly','Quarterly','Monthly'] else 0,
            key='frequency_select',
            help="Time period granularity for the forecast"
        )
        
        # Update session state with the selected value
        st.session_state['frequency'] = frequency
    with col2:
        start_year = st.number_input(
            '📆 Start Year',
            2024, 2030, 2025,
            key='start_year',
            help="First year of the forecast period"
        )
    with col3:
        end_year = st.number_input(
            '📆 End Year',
            2024, 2040, 2030,
            key='end_year',
            help="Last year of the forecast period"
        )

    st.markdown("---")
    
    col_btn1, col_btn2 = st.columns([1, 3])
    with col_btn1:
        if st.button('🚀 Parse & Continue', type='primary', use_container_width=True):
            if not input_text.strip():
                st.error("⚠️ Please enter disease data before continuing")
            else:
                with st.spinner("🔄 Parsing input data..."):
                    parsed = parse_input_v2(input_text)
                    st.session_state['parsed'] = parsed
                    # mark progress
                    st.session_state['step'] = 'parsed'
                st.markdown("""
                <div class="success-card">
                    <strong>✅ Input parsed successfully!</strong><br>
                    You can now proceed to the tabs below to:
                    <ul>
                        <li>Run AI Sub-Agent for automatic parameter estimation</li>
                        <li>Generate base epidemiology forecast</li>
                        <li>Add custom metrics and scenarios</li>
                    </ul>
                </div>
                """, unsafe_allow_html=True)
    # ===== SHOW LOADED MODEL STATUS =====
    if 'model_loaded_from_file' in st.session_state and st.session_state.get('model_loaded_from_file'):
        st.markdown("""
        <div style="background: linear-gradient(135deg, #667eea 0%, #764ba2 100%); 
                    padding: 20px; 
                    border-radius: 10px; 
                    border-left: 5px solid #4CAF50;
                    margin: 20px 0;">
            <h3 style="color: white; margin: 0;">✅ Model Loaded Successfully!</h3>
            <p style="color: white; margin: 10px 0 0 0;">
                Your previously saved model has been loaded. All forecasts, configurations, and custom metrics are now available in the tabs below.
                <br><br>
                <strong>You can now:</strong>
                <ul style="margin: 10px 0 0 20px;">
                    <li>View your loaded forecasts in the tabs below</li>
                    <li>Modify any parameters (prevalence rate, adherence, etc.)</li>
                    <li>Add or remove custom metrics</li>
                    <li>Regenerate forecasts with updated parameters</li>
                    <li>Apply market or brand events</li>
                </ul>
            </p>
        </div>
        """, unsafe_allow_html=True)
        
        # Show quick stats
        col1, col2, col3, col4 = st.columns(4)
        
        with col1:
            if 'df_metrics_base' in st.session_state:
                st.metric("📊 Epi Metrics Loaded", len(st.session_state['df_metrics_base']))
        
        with col2:
            if 'df_brand_forecast' in st.session_state:
                st.metric("💊 Brand Data Loaded", "Yes")
            else:
                st.metric("💊 Brand Data Loaded", "No")
        
        with col3:
            if 'ai_assumptions_v2' in st.session_state:
                custom_count = len(st.session_state['ai_assumptions_v2'].get('derived_metrics', {}))
                st.metric("🎯 Custom Metrics", custom_count)
        
        with col4:
            if 'df_metrics_base' in st.session_state:
                years = [col for col in st.session_state['df_metrics_base'].columns if str(col).isdigit()]
                if years:
                    st.metric("📅 Forecast Years", f"{min(years)}-{max(years)}")
        
        st.markdown("---")
        
        # Add navigation helper  <---- PUT IT HERE
        st.markdown("""
        <div class="info-card">
            <strong>👇 Scroll down to view your loaded data in the tabs below:</strong>
            <ul>
                <li><strong>🔬 Base Forecast</strong> - View and modify epidemiology data</li>
                <li><strong>🎯 Custom Metrics</strong> - View/edit your custom metrics</li>
                <li><strong>💊 Brand Forecast</strong> - View/edit brand configuration and forecasts</li>
            </ul>
        </div>
        """, unsafe_allow_html=True)
        
        # Add a button to clear the loaded model and start fresh
        if st.button("🔄 Clear Loaded Model & Start Fresh", help="Remove loaded model and start with blank inputs"):
            keys_to_remove = ['model_loaded_from_file', 'df_metrics_base', 'df_metrics_custom', 
                            'df_brand_forecast', 'brand_config', 'ai_assumptions_v2', 
                            'parsed', 'input_text', 'step']
            for key in keys_to_remove:
                if key in st.session_state:
                    del st.session_state[key]
            st.rerun()
        
        st.markdown("---")
    # if not parsed yet, show instruction and stop
    if 'parsed' not in st.session_state:
        st.markdown("""
        <div class="info-card">
            <strong>👋 Welcome!</strong><br><br>
            To get started:
            <ol>
                <li>Enter your disease data in the text area above</li>
                <li>Or click one of the "Example" buttons to load sample data</li>
                <li>Configure your forecast timeframe</li>
                <li>Click "🚀 Parse & Continue" to proceed</li>
            </ol>
        </div>
        """, unsafe_allow_html=True)
        return

    parsed = st.session_state['parsed']
    
    st.markdown("---")
    st.markdown('<div class="section-header"><span class="section-icon">📊</span><h2>Step 2: Generate & Customize Forecast</h2></div>', unsafe_allow_html=True)

    # Tabs representing the workflow stages with improved labels
    tab_base, tab_funnel, tab_custom, tab_citations, tab_brand, tab_sensitivity, tab_scenarios = st.tabs([
        "Base Forecast",
        "AI Suggestions",
        "Custom Metrics",
        "Citations & Sources",
        "Brand Analysis",
        "Sensitivity Analysis",
        "Scenarios"
    ])

    # helper used in tab controls
    def _safe_fraction(kv, key, default_frac):
        v = kv.get(key, default_frac)
        try:
            return float(v)
        except Exception:
            if v is None:
                return default_frac
            s = str(v)
            pct_match = re.search(r'([-+]?\d*\.?\d+)\s*%', s)
            if pct_match:
                try:
                    return float(pct_match.group(1))/100.0
                except Exception:
                    pass
            num_match = re.search(r'([-+]?\d*\.?\d+)', s)
            if num_match:
                try:
                    return float(num_match.group(1))
                except Exception:
                    pass
            return default_frac

    # ---------- Tab 1: Base Forecast ----------
    with tab_base:
        st.markdown('<div class="section-header"><span class="section-icon">🔬</span><h3>Generate Base Epidemiology Forecast</h3></div>', unsafe_allow_html=True)
        
        # Show parsed input in collapsible section
        with st.expander("📋 View Parsed Input Data", expanded=False):
            st.json(parsed)
        
        st.markdown("""
        <div class="info-card">
            <strong>Choose your approach:</strong><br>
            • <strong>AI Sub-Agent:</strong> Let AI analyze your input and suggest evidence-based parameters with citations<br>
            • <strong>Sample Data:</strong> Use pre-configured test data for a quick demo
        </div>
        """, unsafe_allow_html=True)

        col1, col2 = st.columns(2)
        with col1:
            if st.button('🤖 Run AI Sub-Agent', type='primary', use_container_width=True, help="Uses AI to estimate epidemiological parameters with research citations"):
                with st.spinner('🔄 Calling AI sub-agent... This may take 10-30 seconds'):
                    try:
                        ai = build_ai_forecast_agent_v2(parsed)
                        st.session_state['ai_assumptions_v2'] = ai
                        st.session_state['step'] = 'ai_run'
                        st.markdown("""
                        <div class="success-card">
                            <strong>✅ AI analysis complete!</strong><br>
                            Review and edit the base rates below, then generate your forecast.
                        </div>
                        """, unsafe_allow_html=True)
                    except Exception as e:
                        st.error(f"⚠️ AI agent error: {str(e)}")
                        st.info("💡 Try using 'Test with Sample Data' instead")
        
        with col2:
            if st.button('🧪 Test with Sample Data', use_container_width=True, help="Load pre-configured example data"):
                # Provide hardcoded reasonable test data
                test_data = {
                    'metrics': ['Total Population', 'Prevalence Rate', 'Prevalent Population', 
                              'Diagnosed Rate', 'Diagnosed Population', 'Treated Rate', 'Treated Population'],
                    'base_year': 2025,
                    'growth_rates': {'Population': 0.007},  # 0.7% annual growth
                    'population': parsed.get('population', 1000000),
                    'known_values': {
                        'prevalence': 0.36,  # 36%
                        'diagnosed': 0.85,    # 85%
                        'treated': 0.78       # 78%
                    },
                    'assumptions': 'Test data with reasonable epidemiology metrics',
                    'citations': {
                        'prevalence': 'Sample data - no citation',
                        'diagnosed': 'Sample data - no citation',
                        'treated': 'Sample data - no citation'
                    }
                }
                st.session_state['ai_assumptions_v2'] = test_data
                st.session_state['step'] = 'ai_run'
                st.markdown("""
                <div class="success-card">
                    <strong>✅ Sample data loaded!</strong><br>
                    You can now edit rates and generate a forecast.
                </div>
                """, unsafe_allow_html=True)

        ai = st.session_state.get('ai_assumptions_v2')
        if ai:
            st.markdown("---")
            st.markdown('<div class="section-header"><span class="section-icon">⚙️</span><h3>Configure Base Parameters</h3></div>', unsafe_allow_html=True)
            
            st.markdown("""
            <div class="info-card">
                <strong>💡 Tips:</strong>
                <ul>
                    <li>Gray boxes indicate editable values - these are your base assumptions</li>
                    <li>Adjust these rates based on your disease and geography</li>
                    <li>All rates are shown as percentages for easier editing</li>
                </ul>
            </div>
            """, unsafe_allow_html=True)
            
            prevalence_frac = _safe_fraction(ai.get('known_values', {}), 'prevalence', 0.02)
            diagnosed_frac = _safe_fraction(ai.get('known_values', {}), 'diagnosed', 0.8)
            treated_frac = _safe_fraction(ai.get('known_values', {}), 'treated', 0.7)

            c1, c2 = st.columns(2)
            with c1:
                prevalence = st.number_input(
                    '📊 Prevalence Rate (%)',
                    value=prevalence_frac*100.0,
                    min_value=0.0,
                    max_value=100.0,
                    step=0.01,
                    key='base_prev',
                    help="Percentage of total population with the disease"
                )
                diagnosed = st.number_input(
                    '🔍 Diagnosed Rate (% of prevalent)',
                    value=diagnosed_frac*100.0,
                    min_value=0.0,
                    max_value=100.0,
                    step=0.1,
                    key='base_diag',
                    help="Percentage of prevalent population that has been diagnosed"
                )
            with c2:
                treated = st.number_input(
                    '💊 Treatment Rate (% of diagnosed)',
                    value=treated_frac*100.0,
                    min_value=0.0,
                    max_value=100.0,
                    step=0.1,
                    key='base_treat',
                    help="Percentage of diagnosed patients receiving treatment"
                )
                # Get growth rate value safely
                growth_rate_val = ai.get('growth_rates', {}).get('Population', 0.01)
                safe_growth_val = growth_rate_val if growth_rate_val is not None else 0.01
                
                pop_growth = st.number_input(
                    '📈 Population Growth (% annual)',
                    value=float(safe_growth_val) * 100.0,
                    step=0.01,
                    key='base_pop_growth',
                    help="Annual population growth rate"
                )

            ai['known_values']['prevalence'] = prevalence/100.0
            ai['known_values']['diagnosed'] = diagnosed/100.0
            ai['known_values']['treated'] = treated/100.0
            ai['growth_rates']['Population'] = pop_growth/100.0
            st.session_state['ai_assumptions_v2'] = ai

            # Display citations
            # Display citations
            citations = ai.get('citations', {})
            if citations:
                with st.expander("📚 View Research Citations & Sources", expanded=False):
                    st.markdown("### Evidence Base for Epidemiological Rates")
                    
                    # Separate base rates from custom metrics/segments
                    base_rates = {}
                    custom_items = {}
                    
                    for metric, citation in citations.items():
                        metric_lower = metric.lower()
                        if metric_lower in ['prevalence', 'diagnosed', 'treated', 'incidence']:
                            base_rates[metric] = citation
                        else:
                            custom_items[metric] = citation
                    
                    # Display base epidemiological rates
                    if base_rates:
                        st.markdown("#### 📊 Base Epidemiological Rates")
                        for metric, citation in base_rates.items():
                            metric_label = metric.replace('_', ' ').title()
                            st.markdown(f"**{metric_label}:**")
                            st.info(citation)
                            st.markdown("---")
                    
                    # Display custom metrics and segments
                    if custom_items:
                        st.markdown("#### 🎯 Custom Metrics & Segments")
                        for metric, citation in custom_items.items():
                            st.markdown(f"**{metric}:**")
                            st.info(citation)
                            st.markdown("---")

            st.markdown("---")
            if st.button('📊 Generate Base Forecast', type='primary', use_container_width=True):
                with st.spinner('🔄 Generating base forecast... This may take a moment'):
                    try:
                        df_base = generate_forecast_v2(ai, start_year, end_year, frequency.lower())
                        st.session_state['df_metrics_base'] = df_base
                        st.session_state['step'] = 'base_generated'
                        st.markdown("""
                        <div class="success-card">
                            <strong>✅ Base forecast generated successfully!</strong><br>
                            Next steps:
                            <ul>
                                <li>Review the forecast table below</li>
                                <li>Go to "AI Suggestions" tab for refinements</li>
                                <li>Add custom metrics if needed</li>
                                <li>Generate brand forecast when ready</li>
                            </ul>
                        </div>
                        """, unsafe_allow_html=True)
                    except Exception as e:
                        st.error(f"⚠️ Error generating forecast: {str(e)}")

        # show base forecast if present
        if 'df_metrics_base' in st.session_state:
            st.markdown("---")
            st.markdown('<div class="section-header"><span class="section-icon">📊</span><h3>Base Forecast Results</h3></div>', unsafe_allow_html=True)
            
            dfb = st.session_state['df_metrics_base']
            dfb_display = dfb.copy()
            
            # Format all COUNT and POPULATION metrics as integers (not percentages)
            count_rows = [r for r in dfb_display.index 
                         if ('population' in r.lower() or 'count' in r.lower()) 
                         and '%' not in r and 'rate' not in r.lower()]
            
            if count_rows:
                try:
                    # Convert to integer display for count metrics
                    dfb_display.loc[count_rows] = dfb_display.loc[count_rows].round(0).astype('Int64')
                except Exception:
                    dfb_display.loc[count_rows] = dfb_display.loc[count_rows].round(0)
            
            # Format all RATE/% metrics as percentages (2 decimal places)
            rate_rows = [r for r in dfb_display.index if '%' in r or 'rate' in r.lower()]
            if rate_rows:
                try:
                    # Keep rates as floats with proper precision
                    dfb_display.loc[rate_rows] = dfb_display.loc[rate_rows].round(4)
                except Exception:
                    pass
            
            st.dataframe(dfb_display, use_container_width=True)
            
            # Show summary metrics
            st.markdown("### 📈 Quick Summary")
            years = list(dfb_display.columns)
            if years:
                first_year = years[0]
                last_year = years[-1]
                
                col1, col2, col3, col4 = st.columns(4)
                
                # Find treated population row
                treated_rows = [r for r in dfb_display.index if 'treated' in r.lower() and 'population' in r.lower() and 'rate' not in r.lower()]
                if treated_rows:
                    treated_row = treated_rows[0]
                    with col1:
                        st.metric(
                            label=f"Treated Population ({first_year})",
                            value=f"{dfb_display.loc[treated_row, first_year]:,.0f}"
                        )
                    with col2:
                        st.metric(
                            label=f"Treated Population ({last_year})",
                            value=f"{dfb_display.loc[treated_row, last_year]:,.0f}",
                            delta=f"{((dfb_display.loc[treated_row, last_year] / dfb_display.loc[treated_row, first_year]) - 1) * 100:.1f}%"
                        )

    # ---------- Tab 2: AI Suggested Funnel Changes ----------
    with tab_funnel:
        st.markdown('<div class="section-header"><span class="section-icon">🤖</span><h3>AI-Suggested Metric Refinements</h3></div>', unsafe_allow_html=True)
        
        if 'df_metrics_base' not in st.session_state:
            st.markdown("""
            <div class="info-card">
                <strong>⚠️ Generate base forecast first</strong><br>
                Go to the "🔬 Base Forecast" tab to generate your initial forecast before using AI suggestions.
            </div>
            """, unsafe_allow_html=True)
        else:
            ai = st.session_state.get('ai_assumptions_v2', {})
            
            st.markdown("""
            <div class="info-card">
                <strong>🤖 AI Metric Suggestions</strong><br>
                The AI will analyze your current pipeline and suggest additional granular metrics to enhance your model.
                These suggestions help you capture:
                <ul>
                    <li>Disease severity levels (mild, moderate, severe)</li>
                    <li>Patient subgroups (age, comorbidities)</li>
                    <li>Treatment response categories (responders, non-responders)</li>
                </ul>
            </div>
            """, unsafe_allow_html=True)
            
            if st.button('🔍 Get AI Suggestions', type='primary', use_container_width=True):
                with st.spinner('🔄 AI is analyzing your metrics and suggesting refinements...'):
                    try:
                        suggestions = funnel_agent_v2(parsed, ai)
                        # Filter suggestions to only those that are not already present as metrics
                        filtered = {}
                        existing = set([m.lower() for m in ai.get('metrics',[])])
                        for k,v in (suggestions or {}).items():
                            # k is like 'Source->NewMetric'
                            parts = k.split('->')
                            if len(parts) == 2:
                                new_metric = parts[1].strip()
                                if new_metric.lower() not in existing:
                                    filtered[k] = v
                        st.session_state['metric_suggestions'] = filtered
                        
                        if filtered:
                            st.markdown("""
                            <div class="success-card">
                                <strong>✅ AI suggestions ready!</strong><br>
                                Review the suggested metrics below and add them in the "➕ Custom Metrics" tab.
                            </div>
                            """, unsafe_allow_html=True)
                        else:
                            st.info("ℹ️ No new metrics suggested - your current pipeline is comprehensive!")
                    except Exception as e:
                        st.error(f"⚠️ AI suggestion error: {str(e)}")

            if st.session_state.get('metric_suggestions'):
                st.markdown("---")
                st.markdown('<div class="section-header"><span class="section-icon">💡</span><h3>Suggested Metrics</h3></div>', unsafe_allow_html=True)
                
                for k,v in st.session_state['metric_suggestions'].items():
                    parts = k.split('->')
                    if len(parts) == 2:
                        source = parts[0].strip()
                        new_metric = parts[1].strip()
                        st.markdown(f"""
                        <div style='background-color: #f0f9ff; padding: 1rem; margin: 0.5rem 0; border-radius: 8px; border-left: 4px solid #3b82f6;'>
                            <strong>📊 {new_metric}</strong><br>
                            <em>From: {source}</em><br>
                            <span style='color: #059669; font-weight: 600;'>Typical Rate: {v:.1%}</span>
                        </div>
                        """, unsafe_allow_html=True)
                
                st.markdown("""
                <div class="info-card">
                    <strong>Next Step:</strong> Go to the "➕ Custom Metrics" tab to add any of these suggested metrics to your forecast.
                </div>
                """, unsafe_allow_html=True)

    with tab_custom:
        st.markdown('<div class="section-header"><span class="section-icon">➕</span><h3>Add Custom Metrics</h3></div>', unsafe_allow_html=True)
        
        if 'df_metrics_base' not in st.session_state:
            st.markdown("""
            <div class="info-card">
                <strong>⚠️ Generate base forecast first</strong><br>
                Either:
                <ul>
                    <li>Upload a previously saved model using the uploader at the top, OR</li>
                    <li>Generate a new forecast in the "🔬 Base Forecast" tab</li>
                </ul>
            </div>
            """, unsafe_allow_html=True)
        else:
            ai = st.session_state.get('ai_assumptions_v2', {})
            metrics_list = ai.get('metrics', [])
            if not metrics_list:
                st.info('No pipeline metrics available. Run AI Sub-Agent in the Base Forecast tab.')
            else:
                # Show currently added custom metrics
                derived = ai.get('derived_metrics', {})
                if derived:
                    st.markdown('<div class="section-header"><span class="section-icon">📋</span><h4>Active Custom Metrics</h4></div>', unsafe_allow_html=True)
                    
                    # Normalize the derived metrics to ensure consistent structure
                    ai['derived_metrics'] = normalize_derived_metrics(ai['derived_metrics'])
                    derived = ai['derived_metrics']
                    citations = ai.get('citations', {})
                    
                    for metric_name, config in derived.items():
                        col1, col2 = st.columns([5, 1])
                        
                        with col1:
                            rate_type = "Percentage" if config['is_rate'] else "Count"
                            base_rate = config['base_rate']
                            growth_rate = config['growth_rate']
                            source = config['source']
                            
                            st.markdown(f"""
                            <div style='background-color: #f0fdf4; padding: 1rem; margin: 0.5rem 0; border-radius: 8px; border-left: 4px solid #10b981;'>
                                <strong>{metric_name}</strong> <span style='color: #059669;'>({rate_type})</span><br>
                                <em>After: {source}</em><br>
                                Base: <strong>{base_rate*100:.1f}%</strong> | Growth: <strong>{growth_rate*100:.2f}%/year</strong>
                            </div>
                            """, unsafe_allow_html=True)
                            
                            # Show citation if available
                            metric_base = metric_name.replace(' %', '').replace(' Count', '').replace(' Rate', '').replace(' Population', '').lower()
                            if metric_base in citations:
                                st.caption(f"📚 Source: {citations[metric_base]}")
                        
                        with col2:
                            if st.button('🗑️ Remove', key=f'delete_{metric_name}', use_container_width=True):
                                del ai['derived_metrics'][metric_name]
                                ai['metrics'] = [m for m in ai['metrics'] if m != metric_name]
                                
                                # Also remove citation if this was the last metric using it
                                metric_base = metric_name.replace(' %', '').replace(' Count', '').replace(' Rate', '').replace(' Population', '').lower()
                                related_metrics = [m for m in ai['metrics'] if metric_base in m.lower()]
                                if not related_metrics and metric_base in ai.get('citations', {}):
                                    del ai['citations'][metric_base]
                                
                                st.session_state['ai_assumptions_v2'] = ai
                                st.success(f"✅ Deleted '{metric_name}'")
                                st.rerun()                    
                    # Add button to clear all custom metrics
                    if st.button('🗑️ Clear All Custom Metrics', type='secondary'):
                        # Get list of custom metrics to remove
                        custom_metric_names = list(ai.get('derived_metrics', {}).keys())
                        
                        # Remove from derived_metrics
                        ai['derived_metrics'] = {}
                        
                        # Remove from metrics list
                        ai['metrics'] = [m for m in ai['metrics'] if m not in custom_metric_names]
                        
                        # Clean up citations for removed custom metrics
                        if 'citations' in ai:
                            for metric_name in custom_metric_names:
                                metric_base = metric_name.replace(' %', '').replace(' Count', '').replace(' Rate', '').replace(' Population', '').lower()
                                if metric_base in ai['citations']:
                                    del ai['citations'][metric_base]
                        
                        st.session_state['ai_assumptions_v2'] = ai
                        st.success('✅ All custom metrics cleared')
                        st.rerun()
                    
                    st.markdown("---")
                
                st.markdown('<div class="section-header"><span class="section-icon">➕</span><h4>Add New Metrics</h4></div>', unsafe_allow_html=True)
                
                # Add tabs for Linear vs Segments
                metric_type_tab = st.tabs(["Add Linear Metric", "Add Segments"])
                
                # ========== TAB 1: LINEAR METRIC (EXISTING) ==========
                with metric_type_tab[0]:
                    st.markdown("""
                    <div class="info-card">
                        <strong>💡 Linear Metrics:</strong><br>
                        Add a single metric that flows linearly through the funnel:
                        <ul>
                            <li><strong>Percentage (%):</strong> A percentage (e.g., "% of treated patients who are severe")</li>
                            <li><strong>Count:</strong> Calculated count (source × percentage)</li>
                            <li><strong>Both:</strong> Creates percentage + count pair for complete modeling</li>
                        </ul>
                    </div>
                        """, unsafe_allow_html=True)
                    
                    # Refresh the metrics list to show current state
                    current_metrics = ai.get('metrics', [])
                    
                    col1, col2 = st.columns(2)
                    
                    with col1:
                        insert_after = st.selectbox(
                            '📍 Insert After Metric',
                            options=current_metrics,
                            key='insert_after_custom',
                            help="The new metric will be added after this metric in the pipeline"
                        )
                        new_metric_base_name = st.text_input(
                            '📝 Metric Name',
                            key='new_metric_custom',
                            placeholder='e.g., "Severe Cases", "Hospitalized"',
                            help="Base name for the metric (Rate/Population will be added automatically)"
                        )
                        
                    with col2:
                        base_pct = st.number_input(
                            '📊 Base Rate (% of source)',
                            value=10.0,
                            min_value=0.0,
                            max_value=100.0,
                            step=0.1,
                            key='new_metric_base_pct_custom',
                            help="What percentage of the source population has this characteristic?"
                        )
                        growth_pct = st.number_input(
                            '📈 Annual Growth Rate (%)',
                            value=0.0,
                            step=0.01,
                            key='new_metric_growth_pct_custom',
                            help="Annual change in this rate (0 = constant rate)"
                        )
                    
                    # Citation input for custom metrics
                    citation_input = st.text_area(
                        '📚 Citation/Source (optional but recommended)',
                        placeholder='e.g., Smith et al., 2023, Journal of Epidemiology, DOI: 10.1234/example',
                        key='custom_metric_citation',
                        help='Provide the source for this rate to maintain evidence-based forecasting',
                        height=80
                    )
                    
                    add_both = st.checkbox(
                        '✅ Add both Percentage (%) AND Count metrics (recommended)',
                        value=True,
                        key='add_both_custom',
                        help="Creates a complete pair: percentage metric + calculated count metric"
                    )
                    
                    if add_both:
                        st.markdown("""
                        <div style='background-color: #dbeafe; padding: 1rem; border-radius: 8px; margin: 0.5rem 0;'>
                            <strong>📊 This will create TWO metrics:</strong><br>
                            1. <strong>[Metric Name] %</strong> - the percentage<br>
                            2. <strong>[Metric Name] Count</strong> - calculated as source × percentage
                        </div>
                        """, unsafe_allow_html=True)
                    else:
                        metric_type = st.radio(
                            'Metric type',
                            ['Percentage only', 'Count only'],
                            key='metric_type_custom',
                            help="Choose whether to add just a percentage or just a count metric"
                        )
                    
                    st.markdown("---")
                    if st.button('➕ Add Linear Metric', type='primary', use_container_width=True, key='add_linear_metric_btn'):
                        if not new_metric_base_name.strip():
                            st.error("⚠️ Please enter a metric name")
                        else:
                            # Validate citation if provided
                            if citation_input.strip():
                                is_valid, message = validate_citation(citation_input)
                                if not is_valid:
                                    st.warning(f"⚠️ Citation validation: {message}")
                                    if not st.checkbox('Add anyway without proper citation', key='bypass_citation'):
                                        st.stop()
                            
                            # Clean the metric name
                            base_name = new_metric_base_name.strip()
                            
                            if add_both:
                                # Add BOTH percentage and count metrics
                                rate_name = f"{base_name} %"
                                pop_name = f"{base_name} Count"
                                
                                # Check if they already exist
                                if rate_name in ai.get('metrics', []) or pop_name in ai.get('metrics', []):
                                    st.error(f"⚠️ One or both metrics already exist!")
                                else:
                                    # Find the index to insert after
                                    insert_idx = current_metrics.index(insert_after) + 1
                                    
                                    # Initialize derived_metrics if needed
                                    if 'derived_metrics' not in ai:
                                        ai['derived_metrics'] = {}
                                    
                                    # Add RATE metric first
                                    ai['metrics'].insert(insert_idx, rate_name)
                                    ai['derived_metrics'][rate_name] = {
                                        'source': insert_after,
                                        'base_rate': base_pct/100.0,
                                        'growth_rate': growth_pct/100.0,
                                        'is_rate': True  # This is a rate
                                    }
                                    
                                    # Add POPULATION metric right after rate
                                    ai['metrics'].insert(insert_idx + 1, pop_name)
                                    ai['derived_metrics'][pop_name] = {
                                        'source': insert_after,  # Source is the original metric (e.g., Treated Population)
                                        'rate_metric': rate_name,  # Link to the corresponding rate metric
                                        'base_rate': base_pct/100.0,
                                        'growth_rate': growth_pct/100.0,
                                        'is_rate': False  # This is a population count
                                    }
                                    
                                    # Store citation for custom metric
                                    if citation_input.strip():
                                        if 'citations' not in ai:
                                            ai['citations'] = {}
                                        ai['citations'][base_name.lower()] = citation_input.strip()
                                    
                                    st.session_state['ai_assumptions_v2'] = ai
                                    st.markdown(f"""
                                    <div class="success-card">
                                        <strong>✅ Metrics added successfully!</strong><br>
                                        Added: <strong>{rate_name}</strong> and <strong>{pop_name}</strong> after <em>{insert_after}</em>
                                    </div>
                                    """, unsafe_allow_html=True)
                                    if citation_input.strip():
                                        st.info(f"📚 Citation saved")
                                    st.rerun()
                            else:
                                # Add single metric (old behavior)
                                is_rate = (metric_type == 'Percentage only')
                                metric_name = f"{base_name} %" if is_rate else f"{base_name} Count"
                                
                                if metric_name in ai.get('metrics', []):
                                    st.error(f"⚠️ Metric '{metric_name}' already exists!")
                                else:
                                    insert_idx = current_metrics.index(insert_after) + 1
                                    
                                    ai['metrics'].insert(insert_idx, metric_name)
                                    
                                    if 'derived_metrics' not in ai:
                                        ai['derived_metrics'] = {}
                                    
                                    ai['derived_metrics'][metric_name] = {
                                        'source': insert_after,
                                        'base_rate': base_pct/100.0,
                                        'growth_rate': growth_pct/100.0,
                                        'is_rate': is_rate
                                    }
                                    
                                    # Store citation
                                    if citation_input.strip():
                                        if 'citations' not in ai:
                                            ai['citations'] = {}
                                        ai['citations'][base_name.lower()] = citation_input.strip()
                                    
                                    st.session_state['ai_assumptions_v2'] = ai
                                    st.markdown(f"""
                                    <div class="success-card">
                                        <strong>✅ Metric added successfully!</strong><br>
                                        Added: <strong>{metric_name}</strong> after <em>{insert_after}</em>
                                    </div>
                                    """, unsafe_allow_html=True)
                                    if citation_input.strip():
                                        st.info(f"📚 Citation saved")
                                    st.rerun()
                
                # ========== TAB 2: ADD SEGMENTS (NEW) ==========
                with metric_type_tab[1]:
                    st.markdown("""
                    <div class="info-card">
                        <strong>🌳 Segment Branching:</strong><br>
                        Create multiple segments that branch from a parent population:
                        <ul>
                            <li><strong>Example:</strong> Split "Diagnosed Population" into Primary, Secondary, and HeFH segments</li>
                            <li><strong>Each segment gets:</strong> Its own percentage (%) + count metrics</li>
                            <li><strong>Validation:</strong> Segment percentages must sum to 100%</li>
                            <li><strong>Result:</strong> Full patient funnel with branching structure</li>
                        </ul>
                    </div>
                    """, unsafe_allow_html=True)
                    
                    current_metrics_seg = ai.get('metrics', [])
                    
                    # Segment configuration
                    col1_seg, col2_seg = st.columns(2)
                    
                    with col1_seg:
                        segment_parent = st.selectbox(
                            '📍 Parent Metric to Segment',
                            options=current_metrics_seg,
                            key='segment_parent',
                            help="This population will be split into segments"
                        )
                        
                        num_segments = st.number_input(
                            '🔢 Number of Segments',
                            min_value=2,
                            max_value=5,
                            value=3,
                            step=1,
                            key='num_segments',
                            help="How many segments to create (2-5)"
                        )
                    
                    with col2_seg:
                        st.markdown("### Segment Configuration")
                        st.caption("Define each segment below")
                    
                    st.markdown("---")
                    st.markdown("#### 📋 Define Segments")
                    
                    # Initialize segments in session state if not exists
                    if 'temp_segments' not in st.session_state:
                        st.session_state['temp_segments'] = []
                    
                    # Create input fields for each segment
                    segments_data = []
                    total_pct = 0.0
                    
                    for i in range(int(num_segments)):
                        st.markdown(f"**Segment {i+1}**")
                        col_name, col_pct = st.columns([2, 1])
                        
                        with col_name:
                            temp_segs = st.session_state.get('temp_segments', [])
                            default_name = f'Segment {chr(65+i)}'
                            seg_name = st.text_input(
                                f'Segment {i+1} Name',
                                value=temp_segs[i].get('name', default_name) if i < len(temp_segs) else default_name,
                                key=f'seg_name_{i}',
                                placeholder=f'e.g., "Primary Prevention", "Secondary Prevention"'
                            )
                        
                        with col_pct:
                            temp_segs = st.session_state.get('temp_segments', [])
                            default_pct = 100.0/num_segments
                            seg_pct = st.number_input(
                                f'% of Parent',
                                min_value=0.0,
                                max_value=100.0,
                                value=temp_segs[i].get('pct', default_pct) if i < len(temp_segs) else default_pct,
                                step=0.1,
                                key=f'seg_pct_{i}',
                                help=f"Percentage of {segment_parent}"
                            )
                        
                        segments_data.append({
                            'name': seg_name,
                            'pct': seg_pct
                        })
                        total_pct += seg_pct
                    
                    # Validation display
                    st.markdown("---")
                    if abs(total_pct - 100.0) < 0.01:
                        st.success(f"✅ Total: {total_pct:.1f}% (Valid!)")
                    else:
                        st.error(f"⚠️ Total: {total_pct:.1f}% (Must equal 100%!)")
                    
                    # Citation for segments
                    segment_citation = st.text_area(
                        '📚 Citation/Source for Segment Distribution (optional)',
                        placeholder='e.g., Clinical guidelines showing patient distribution across risk categories',
                        key='segment_citation',
                        height=80
                    )
                    
                    st.markdown("---")
                    if st.button('🌳 Create Segments', type='primary', use_container_width=True, key='add_segments_btn'):
                        # Validate total percentage
                        if abs(total_pct - 100.0) >= 0.01:
                            st.error("⚠️ Segment percentages must sum to 100%!")
                        else:
                            # Validate segment names
                            seg_names = [s['name'].strip() for s in segments_data]
                            if not all(seg_names):
                                st.error("⚠️ All segments must have names!")
                            elif len(seg_names) != len(set(seg_names)):
                                st.error("⚠️ Segment names must be unique!")
                            else:
                                # Create the segments
                                insert_idx = current_metrics_seg.index(segment_parent) + 1
                                
                                # Initialize structures
                                if 'derived_metrics' not in ai:
                                    ai['derived_metrics'] = {}
                                if 'segments' not in ai:
                                    ai['segments'] = {}
                                
                                # Store segment group info
                                segment_group_id = f"{segment_parent}_segments"
                                ai['segments'][segment_group_id] = {
                                    'parent': segment_parent,
                                    'children': []
                                }
                                
                                # Add each segment (rate + population)
                                added_metrics = []
                                for seg in segments_data:
                                    seg_name = seg['name'].strip()
                                    seg_rate = seg['pct'] / 100.0
                                    
                                    # Create rate and population metric names
                                    rate_metric_name = f"{seg_name} %"
                                    pop_metric_name = f"{seg_name} Count"
                                    
                                    # Add RATE metric
                                    ai['metrics'].insert(insert_idx, rate_metric_name)
                                    ai['derived_metrics'][rate_metric_name] = {
                                        'source': segment_parent,
                                        'base_rate': seg_rate,
                                        'growth_rate': 0.0,  # Segments have fixed distribution
                                        'is_rate': True,
                                        'is_segment': True,
                                        'segment_group': segment_group_id
                                    }
                                    insert_idx += 1
                                    added_metrics.append(rate_metric_name)
                                    
                                    # Add POPULATION metric
                                    ai['metrics'].insert(insert_idx, pop_metric_name)
                                    ai['derived_metrics'][pop_metric_name] = {
                                        'source': segment_parent,
                                        'base_rate': seg_rate,
                                        'growth_rate': 0.0,
                                        'is_rate': False,
                                        'is_segment': True,
                                        'segment_group': segment_group_id,
                                        'rate_metric': rate_metric_name  # Link to corresponding rate
                                    }
                                    insert_idx += 1
                                    added_metrics.append(pop_metric_name)
                                    
                                    # Track segment children
                                    ai['segments'][segment_group_id]['children'].append({
                                        'name': seg_name,
                                        'rate_metric': rate_metric_name,
                                        'pop_metric': pop_metric_name,
                                        'percentage': seg_rate * 100
                                    })
                                
                                # Store citation if provided
                                if segment_citation.strip():
                                    if 'citations' not in ai:
                                        ai['citations'] = {}
                                    ai['citations'][f'{segment_parent.lower()}_segments'] = segment_citation.strip()
                                
                                st.session_state['ai_assumptions_v2'] = ai
                                st.session_state['temp_segments'] = []  # Clear temp data
                                
                                st.markdown(f"""
                                <div class="success-card">
                                    <strong>✅ Segments created successfully!</strong><br>
                                    Created {len(segments_data)} segments from <em>{segment_parent}</em>:<br>
                                    {'<br>'.join([f"• {m}" for m in added_metrics])}
                                </div>
                                """, unsafe_allow_html=True)
                                if segment_citation.strip():
                                    st.info("📚 Citation saved")
                                st.rerun()

                st.markdown("---")
                if st.button('📊 Generate Forecast with Custom Metrics', type='primary', use_container_width=True):
                    with st.spinner('🔄 Generating forecast with custom metrics...'):
                        try:
                            df_custom = generate_forecast_v2(ai, start_year, end_year, frequency.lower())
                            st.session_state['df_metrics_custom'] = df_custom
                            st.markdown("""
                            <div class="success-card">
                                <strong>✅ Custom forecast generated successfully!</strong><br>
                                See the results in the table below. You can now proceed to the Brand Analysis tab.
                            </div>
                            """, unsafe_allow_html=True)
                        except Exception as e:
                            st.error(f"⚠️ Error generating forecast: {str(e)}")

                if 'df_metrics_custom' in st.session_state:
                    st.markdown("---")
                    st.markdown('<div class="section-header"><span class="section-icon">📊</span><h3>Custom Forecast Results</h3></div>', unsafe_allow_html=True)
                    
                    dfc = st.session_state['df_metrics_custom']
                    dfc_display = dfc.copy()
                    
                    # Add tree structure visualization for segments
                    segments_info = ai.get('segments', {})
                    if segments_info:
                        # Create a new index with indentation for segments
                        new_index = []
                        for metric in dfc_display.index:
                            is_segment = False
                            indent = ""
                            
                            # Check if this metric is part of a segment group
                            for segment_group_id, segment_data in segments_info.items():
                                for child in segment_data.get('children', []):
                                    if metric == child['rate_metric'] or metric == child['pop_metric']:
                                        is_segment = True
                                        indent = "  ├─ "  # Tree branch symbol
                                        break
                                if is_segment:
                                    break
                            
                            # Add indentation to metric name
                            new_index.append(f"{indent}{metric}")
                        
                        dfc_display.index = new_index
                    
                    # Format all COUNT and POPULATION metrics as integers (not percentages)
                    count_rows = [r for r in dfc_display.index 
                                 if ('population' in r.lower() or 'count' in r.lower()) 
                                 and '%' not in r and 'rate' not in r.lower()]
                    
                    if count_rows:
                        try:
                            # Convert to integer display for count metrics
                            dfc_display.loc[count_rows] = dfc_display.loc[count_rows].round(0).astype('Int64')
                        except Exception:
                            dfc_display.loc[count_rows] = dfc_display.loc[count_rows].round(0)
                    
                    # Format all RATE/% metrics as percentages (4 decimal places for precision)
                    rate_rows = [r for r in dfc_display.index if '%' in r or 'rate' in r.lower()]
                    if rate_rows:
                        try:
                            # Keep rates as floats with proper precision
                            dfc_display.loc[rate_rows] = dfc_display.loc[rate_rows].round(4)
                        except Exception:
                            pass
                    
                    st.dataframe(dfc_display, use_container_width=True)

    # ---------- Tab 4: Citations ----------
    with tab_citations:
        st.markdown('<div class="section-header"><span class="section-icon">📚</span><h3>Research Citations & Evidence Base</h3></div>', unsafe_allow_html=True)
        
        ai = st.session_state.get('ai_assumptions_v2', {})
        citations = ai.get('citations', {})
        
        if not citations:
            st.markdown("""
            <div class="info-card">
                <strong>📚 No citations available yet</strong><br>
                Generate a forecast first by:
                <ul>
                    <li>Running the AI Sub-Agent in the Base Forecast tab</li>
                    <li>Using Test Data to quickly populate the model</li>
                    <li>Adding custom metrics with citations</li>
                </ul>
                Citations help ensure your forecast is evidence-based and transparent.
            </div>
            """, unsafe_allow_html=True)
        else:
            st.markdown("""
            <div class="info-card">
                <strong>📖 About Citations</strong><br>
                This page shows all research sources and citations for epidemiological rates used in your forecast.
                Citations ensure transparency, reproducibility, and evidence-based decision making.
            </div>
            """, unsafe_allow_html=True)
            
            # Base rates citations
            st.markdown("---")
            st.markdown('<div class="section-header"><span class="section-icon">🔬</span><h4>Base Epidemiological Rates</h4></div>', unsafe_allow_html=True)
            
            known_values = ai.get('known_values', {})
            
            for metric in ['prevalence', 'diagnosed', 'treated']:
                if metric in known_values:
                    col1, col2 = st.columns([1, 3])
                    
                    with col1:
                        st.metric(
                            label=metric.title(),
                            value=f"{known_values[metric]*100:.2f}%"
                        )
                    
                    with col2:
                        if metric in citations:
                            st.markdown(f"**Source:**")
                            st.info(citations[metric])
                        else:
                            st.warning("No citation provided")
                    
                    st.markdown("---")
            
            # Custom metrics citations
            derived = ai.get('derived_metrics', {})
            if derived:
                st.subheader('Custom Metrics Citations')
                
                # Get unique base names from custom metrics
                custom_citations = {}
                for metric_name, config in derived.items():
                    metric_base = metric_name.replace(' %', '').replace(' Count', '').replace(' Rate', '').replace(' Population', '').lower()
                    
                    if metric_base in citations and metric_base not in custom_citations:
                        custom_citations[metric_base] = {
                            'rate': config['base_rate'],
                            'citation': citations[metric_base]
                        }
                
                for metric_base, info in custom_citations.items():
                    col1, col2 = st.columns([1, 3])
                    
                    with col1:
                        st.metric(
                            label=metric_base.replace('_', ' ').title(),
                            value=f"{info['rate']*100:.2f}%"
                        )
                    
                    with col2:
                        st.markdown(f"**Source:**")
                        st.info(info['citation'])
                    
                    st.markdown("---")
            
            # Export citations as text
            st.subheader('Export Citations')
            
            citations_text = "# Research Citations\n\n"
            citations_text += f"Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n\n"
            
            citations_text += "## Base Epidemiological Rates\n\n"
            for metric in ['prevalence', 'diagnosed', 'treated']:
                if metric in known_values and metric in citations:
                    citations_text += f"**{metric.title()}** ({known_values[metric]*100:.2f}%):\n"
                    citations_text += f"{citations[metric]}\n\n"
            
            if derived:
                citations_text += "## Custom Metrics\n\n"
                for metric_base, info in custom_citations.items():
                    citations_text += f"**{metric_base.replace('_', ' ').title()}** ({info['rate']*100:.2f}%):\n"
                    citations_text += f"{info['citation']}\n\n"
            
            st.download_button(
                label="Download Citations as Text",
                data=citations_text,
                file_name=f"citations_{datetime.now().strftime('%Y%m%d_%H%M%S')}.txt",
                mime="text/plain"
            )
        # ---------- Tab 5: Brand Forecast ----------
    with tab_brand:
        st.header('💊 Brand/Product Forecast')
        
        if 'df_metrics_custom' in st.session_state:
            df_for_brand = st.session_state['df_metrics_custom']
        elif 'df_metrics_base' in st.session_state:
            df_for_brand = st.session_state['df_metrics_base']
        else:
            st.info('Generate an epidemiology forecast first (Base or Custom).')
            st.stop()
        
        st.markdown("""
        Configure brand-specific parameters to calculate market uptake, patient volumes, and revenue projections.
        The brand forecast will be integrated with your epidemiology model.
        """)
        
        # Brand Configuration
        st.subheader('📋 Brand Configuration')
        
        col1, col2, col3 = st.columns(3)
        
        with col1:
            brand_name = st.text_input('Brand Name', value='Brand X', key='brand_name')
            launch_year = st.number_input('Launch Year', min_value=2020, max_value=2050, value=2026, key='launch_year')
            peak_share = st.number_input('Peak Market Share (%)', min_value=0.0, max_value=100.0, value=35.0, step=1.0, key='peak_share')
        
        with col2:
            # Available populations from forecast - INCLUDE all count metrics (population, custom metrics)
            available_pops = [m for m in df_for_brand.index 
                            if ('population' in m.lower() or 'count' in m.lower()) 
                            and 'rate' not in m.lower() 
                            and '%' not in m]
            
            # If no populations found, use all non-rate metrics as fallback
            if not available_pops:
                available_pops = [m for m in df_for_brand.index if '%' not in m and 'rate' not in m.lower()]
            
            target_population = st.selectbox('Target Population', options=available_pops, 
                                            index=len(available_pops)-1 if available_pops else 0,
                                            key='target_population')
            
            uptake_types = list(UPTAKE_CURVES.keys())
            uptake_type = st.selectbox('Uptake Curve', options=uptake_types, 
                                    index=uptake_types.index('7 Year Medium') if '7 Year Medium' in uptake_types else 0,
                                    key='uptake_type')
            
            adherence = st.number_input('Adherence Rate (%)', min_value=0.0, max_value=100.0, value=68.0, step=1.0, key='adherence')
        
        with col3:
            annual_dose = st.number_input('Annual Doses per Patient', min_value=1, max_value=365, value=360, key='annual_dose')
            price_per_dose = st.number_input('Price per Dose ($)', min_value=0.0, value=16.0, step=1.0, key='price_per_dose')
        
        # Show selected uptake curve
        with st.expander("📈 View Selected Uptake Curve", expanded=False):
            curve_data = UPTAKE_CURVES[uptake_type]
            curve_df = pd.DataFrame({
                'Year': [f'Y{i+1}' for i in range(len(curve_data))],
                'Uptake %': [f'{v*100:.0f}%' for v in curve_data]
            })
            
            col1, col2 = st.columns([2, 1])
            with col1:
                # Plot the curve
                import plotly.graph_objects as go
                fig = go.Figure()
                fig.add_trace(go.Scatter(
                    x=list(range(1, len(curve_data) + 1)),
                    y=[v * 100 for v in curve_data],
                    mode='lines+markers',
                    name=uptake_type,
                    line=dict(color='#70AD47', width=3),
                    marker=dict(size=8)
                ))
                fig.update_layout(
                    title=f'{uptake_type} Uptake Curve',
                    xaxis_title='Years Since Launch',
                    yaxis_title='Uptake (%)',
                    yaxis=dict(range=[0, 105]),
                    height=300
                )
                st.plotly_chart(fig, use_container_width=True)
            
            with col2:
                st.dataframe(curve_df, hide_index=True, height=300)
        def ensure_brand_config_complete():
            """Ensure brand_config has all required keys"""
            if 'brand_config' in st.session_state and st.session_state['brand_config']:
                brand_config = st.session_state['brand_config']
                default_config = {
                    'brand_name': 'Brand X',
                    'launch_year': 2025,
                    'peak_share': 0.15,
                    'uptake_type': '7 Year Medium',
                    'target_population': 'Treated Population',
                    'adherence': 0.68,
                    'annual_dose': 360,
                    'price_per_dose': 16.0
                }
                
                # Fill in any missing keys
                for key, default_value in default_config.items():
                    if key not in brand_config:
                        brand_config[key] = default_value
                
                st.session_state['brand_config'] = brand_config
        ensure_brand_config_complete()        
        # Generate brand forecast
        if st.button('💊 Generate Brand Forecast', type='primary', use_container_width=True):
            brand_config = {
                'brand_name': brand_name,
                'launch_year': launch_year,
                'peak_share': peak_share / 100.0,
                'uptake_type': uptake_type,
                'target_population': target_population,
                'adherence': adherence / 100.0,
                'annual_dose': annual_dose,
                'price_per_dose': price_per_dose
            }
            
            # Get current events from session state
            current_events = st.session_state.get('brand_events', [])
            
            with st.spinner('Generating brand forecast with events...'):
                years = [int(col) for col in df_for_brand.columns if str(col).isdigit()]
                start_year_val = min(years) if years else 2025
                end_year_val = max(years) if years else 2030
                
                # Generate brand forecast with events
                df_brand = generate_brand_forecast(df_for_brand, brand_config, start_year_val, end_year_val, current_events)
                st.session_state['df_brand_forecast'] = df_brand
                st.session_state['brand_config'] = brand_config
                
                if current_events:
                    st.success(f'✅ Brand forecast generated successfully with {len(current_events)} market event(s)!')
                else:
                    st.success('✅ Brand forecast generated successfully!')
        
        # ===== DISPLAY BRAND FORECAST =====
        if 'df_brand_forecast' in st.session_state:
            st.markdown("---")
            st.markdown(f"### 💊 {brand_name} Forecast")
            
            df_brand_display = st.session_state['df_brand_forecast'].copy().astype(object)
            
            # Format brand data with proper volume/dose handling
            for idx in df_brand_display.index:
                # Check for volume/dose FIRST (before percentage checks)
                if 'volume' in idx.lower() or ('dose' in idx.lower() and 'price' not in idx.lower()):
                    df_brand_display.loc[idx] = df_brand_display.loc[idx].apply(lambda x: f'{float(x):,.0f}' if pd.notna(x) and x != '' else '')
                # Then check for rates/shares (but NOT if it's a volume/dose metric)
                elif 'share' in idx.lower() or ('rate' in idx.lower() and 'volume' not in idx.lower()):
                    df_brand_display.loc[idx] = df_brand_display.loc[idx].apply(lambda x: f'{float(x)*100:.2f}%' if pd.notna(x) and x != '' else '')
                # Adherence as percentage (but already caught volume above)
                elif 'adherence' in idx.lower() and 'volume' not in idx.lower():
                    df_brand_display.loc[idx] = df_brand_display.loc[idx].apply(lambda x: f'{float(x)*100:.2f}%' if pd.notna(x) and x != '' else '')
                # Revenue
                elif 'revenue' in idx.lower() or 'billions' in idx.lower():
                    df_brand_display.loc[idx] = df_brand_display.loc[idx].apply(lambda x: f'${float(x):,.1f}B' if pd.notna(x) and x != '' else '')
                # Price
                elif 'price' in idx.lower() or '$' in idx:
                    df_brand_display.loc[idx] = df_brand_display.loc[idx].apply(lambda x: f'${float(x):,.2f}' if pd.notna(x) and x != '' else '')
                # Default: whole numbers for patient counts and volumes
                else:
                    df_brand_display.loc[idx] = df_brand_display.loc[idx].apply(lambda x: f'{float(x):,.0f}' if pd.notna(x) and x != '' else '')
            
            st.dataframe(df_brand_display, use_container_width=True)
            
            # # ===== VISUALIZATIONS =====
            # st.markdown("---")
            # st.markdown("### 📊 Key Metrics Visualization")
            
            # df_brand_viz = st.session_state['df_brand_forecast']
            # year_cols = [col for col in df_brand_viz.columns if str(col).isdigit()]
            
            # if len(year_cols) > 0:
            #     # 1. Revenue Chart
            #     revenue_metric_found = False
            #     for possible_name in ['Revenue in billions', f'{brand_name} Revenue (Billions $)', 'Revenue (Billions $)']:
            #         if possible_name in df_brand_viz.index:
            #             try:
            #                 revenue_data = df_brand_viz.loc[possible_name, year_cols].astype(float)
            #                 revenue_metric_found = True
                            
            #                 fig_rev = go.Figure()
            #                 fig_rev.add_trace(go.Bar(
            #                     x=year_cols,
            #                     y=revenue_data,
            #                     name='Revenue',
            #                     marker_color='#34D399',
            #                     opacity=0.8
            #                 ))
            #                 fig_rev.update_layout(
            #                     title=f'{brand_name} Revenue Projection',
            #                     xaxis_title='Year',
            #                     yaxis_title='Revenue (Billions $)',
            #                     hovermode='x unified',
            #                     height=400
            #                 )
            #                 st.plotly_chart(fig_rev, use_container_width=True)
            #                 break
            #             except Exception as e:
            #                 continue
                
            #     # 2. Patient Volume Chart
            #     patient_metric_found = False
            #     for possible_name in ['Patients on Brand X', f'{brand_name} Patients', 'Brand Patients', f'Patients on {brand_name}']:
            #         if possible_name in df_brand_viz.index:
            #             try:
            #                 patient_data = df_brand_viz.loc[possible_name, year_cols].astype(float)
            #                 patient_metric_found = True
                            
            #                 fig_pat = go.Figure()
            #                 fig_pat.add_trace(go.Scatter(
            #                     x=year_cols,
            #                     y=patient_data,
            #                     mode='lines+markers',
            #                     name='Patients',
            #                     line=dict(color='#60A5FA', width=3),
            #                     marker=dict(size=8),
            #                     fill='tozeroy',
            #                     fillcolor='rgba(96, 165, 250, 0.2)'
            #                 ))
            #                 fig_pat.update_layout(
            #                     title=f'{brand_name} Patient Volume',
            #                     xaxis_title='Year',
            #                     yaxis_title='Patients',
            #                     hovermode='x unified',
            #                     height=400
            #                 )
            #                 st.plotly_chart(fig_pat, use_container_width=True)
            #                 break
            #             except Exception as e:
            #                 continue
                
            #     # 3. Market Share Chart
            #     share_metric_found = False
            #     for possible_name in ['Brand X Patient Share', f'{brand_name} Patient Share', 'Patient Share', 'Market Share']:
            #         if possible_name in df_brand_viz.index:
            #             try:
            #                 share_data = df_brand_viz.loc[possible_name, year_cols].astype(float)
            #                 share_metric_found = True
                            
            #                 fig_share = go.Figure()
            #                 fig_share.add_trace(go.Scatter(
            #                     x=year_cols,
            #                     y=share_data * 100,  # Convert to percentage
            #                     mode='lines+markers',
            #                     name='Market Share',
            #                     line=dict(color='#F59E0B', width=3),
            #                     marker=dict(size=8)
            #                 ))
            #                 fig_share.update_layout(
            #                     title=f'{brand_name} Market Share Evolution',
            #                     xaxis_title='Year',
            #                     yaxis_title='Market Share (%)',
            #                     hovermode='x unified',
            #                     height=400
            #                 )
            #                 st.plotly_chart(fig_share, use_container_width=True)
            #                 break
            #             except Exception as e:
            #                 continue
            
            # # ===== COMPARISON CHARTS IF EVENTS EXIST =====
            # if 'df_brand_with_events' in st.session_state:
            #     st.markdown("---")
            #     st.markdown("### 📊 Impact Analysis: With vs Without Events")
                
            #     df_brand_original = st.session_state['df_brand_forecast']
            #     df_brand_with_events = st.session_state['df_brand_with_events']
                
            #     # 1. Revenue Comparison
            #     revenue_comp_found = False
            #     for possible_name in ['Revenue in billions', f'{brand_name} Revenue (Billions $)', 'Revenue (Billions $)']:
            #         if possible_name in df_brand_original.index and len(year_cols) > 0:
            #             try:
            #                 revenue_original = df_brand_original.loc[possible_name, year_cols].astype(float)
            #                 revenue_events = df_brand_with_events.loc[possible_name, year_cols].astype(float)
            #                 revenue_comp_found = True
                            
            #                 fig_revenue = go.Figure()
            #                 fig_revenue.add_trace(go.Bar(
            #                     x=year_cols,
            #                     y=revenue_original,
            #                     name='Without Events',
            #                     marker_color='#93C5FD',
            #                     opacity=0.7
            #                 ))
            #                 fig_revenue.add_trace(go.Bar(
            #                     x=year_cols,
            #                     y=revenue_events,
            #                     name='With Events',
            #                     marker_color='#34D399',
            #                     opacity=0.7
            #                 ))
            #                 fig_revenue.update_layout(
            #                     title=f'{brand_name} Revenue: Impact of Events',
            #                     xaxis_title='Year',
            #                     yaxis_title='Revenue (Billions $)',
            #                     barmode='group',
            #                     hovermode='x unified',
            #                     height=400
            #                 )
            #                 st.plotly_chart(fig_revenue, use_container_width=True)
            #                 break
            #             except Exception as e:
            #                 continue
                
            #     # 2. Volume Comparison
            #     volume_comp_found = False
            #     for possible_name in ['Total adherence adjusted Volume', 'Total Adherence Adjusted Volume', 'Total Volume']:
            #         if possible_name in df_brand_original.index and len(year_cols) > 0:
            #             try:
            #                 volume_original = df_brand_original.loc[possible_name, year_cols].astype(float)
            #                 volume_events = df_brand_with_events.loc[possible_name, year_cols].astype(float)
            #                 volume_comp_found = True
                            
            #                 fig_volume = go.Figure()
            #                 fig_volume.add_trace(go.Scatter(
            #                     x=year_cols,
            #                     y=volume_original,
            #                     name='Without Events',
            #                     mode='lines+markers',
            #                     line=dict(color='#93C5FD', width=3),
            #                     marker=dict(size=8),
            #                     fill='tozeroy',
            #                     fillcolor='rgba(147, 197, 253, 0.2)'
            #                 ))
            #                 fig_volume.add_trace(go.Scatter(
            #                     x=year_cols,
            #                     y=volume_events,
            #                     name='With Events',
            #                     mode='lines+markers',
            #                     line=dict(color='#34D399', width=3),
            #                     marker=dict(size=8),
            #                     fill='tozeroy',
            #                     fillcolor='rgba(52, 211, 153, 0.2)'
            #                 ))
            #                 fig_volume.update_layout(
            #                     title='Total Volume: Impact of Events',
            #                     xaxis_title='Year',
            #                     yaxis_title='Volume (Doses)',
            #                     hovermode='x unified',
            #                     height=400
            #                 )
            #                 st.plotly_chart(fig_volume, use_container_width=True)
            #                 break
            #             except Exception as e:
            #                 continue
                
            #     # 3. Market Share Comparison
            #     share_comp_found = False
            #     for possible_name in ['Brand X Patient Share', f'{brand_name} Patient Share', 'Patient Share', 'Market Share']:
            #         if possible_name in df_brand_original.index and len(year_cols) > 0:
            #             try:
            #                 share_original = df_brand_original.loc[possible_name, year_cols].astype(float)
            #                 share_events = df_brand_with_events.loc[possible_name, year_cols].astype(float)
            #                 share_comp_found = True
                            
            #                 fig_share_comp = go.Figure()
            #                 fig_share_comp.add_trace(go.Scatter(
            #                     x=year_cols,
            #                     y=share_original * 100,
            #                     name='Without Events',
            #                     mode='lines+markers',
            #                     line=dict(color='#93C5FD', width=3),
            #                     marker=dict(size=8)
            #                 ))
            #                 fig_share_comp.add_trace(go.Scatter(
            #                     x=year_cols,
            #                     y=share_events * 100,
            #                     name='With Events',
            #                     mode='lines+markers',
            #                     line=dict(color='#34D399', width=3),
            #                     marker=dict(size=8)
            #                 ))
            #                 fig_share_comp.update_layout(
            #                     title=f'{brand_name} Market Share: Impact of Events',
            #                     xaxis_title='Year',
            #                     yaxis_title='Market Share (%)',
            #                     hovermode='x unified',
            #                     height=400
            #                 )
            #                 st.plotly_chart(fig_share_comp, use_container_width=True)
            #                 break
            #             except Exception as e:
            #                 continue
                
            #     # 4. Patients Comparison
            #     patients_comp_found = False
            #     for possible_name in ['Patients on Brand X', f'{brand_name} Patients', 'Brand Patients', f'Patients on {brand_name}']:
            #         if possible_name in df_brand_original.index and len(year_cols) > 0:
            #             try:
            #                 patients_original = df_brand_original.loc[possible_name, year_cols].astype(float)
            #                 patients_events = df_brand_with_events.loc[possible_name, year_cols].astype(float)
            #                 patients_comp_found = True
                            
            #                 fig_patients = go.Figure()
            #                 fig_patients.add_trace(go.Scatter(
            #                     x=year_cols,
            #                     y=patients_original,
            #                     name='Without Events',
            #                     mode='lines+markers',
            #                     line=dict(color='#93C5FD', width=3),
            #                     marker=dict(size=8)
            #                 ))
            #                 fig_patients.add_trace(go.Scatter(
            #                     x=year_cols,
            #                     y=patients_events,
            #                     name='With Events',
            #                     mode='lines+markers',
            #                     line=dict(color='#34D399', width=3),
            #                     marker=dict(size=8)
            #                 ))
            #                 fig_patients.update_layout(
            #                     title=f'{brand_name} Patients: Impact of Events',
            #                     xaxis_title='Year',
            #                     yaxis_title='Patients',
            #                     hovermode='x unified',
            #                     height=400
            #                 )
            #                 st.plotly_chart(fig_patients, use_container_width=True)
            #                 break
            #             except Exception as e:
            #                 continue
            
            # # ===== DOWNLOAD BUTTON =====
            # st.markdown("---")
            # st.markdown("### 📥 Export Forecast")
            
            # if st.button('📥 Download Complete Forecast', type='primary', use_container_width=True):
            #     with st.spinner('Preparing Excel export...'):
            #         try:
            #             # Determine which forecasts to export
            #             if 'df_brand_with_events' in st.session_state:
            #                 df_brand_to_export = st.session_state['df_brand_with_events']
            #                 filename_suffix = "_with_events"
            #             else:
            #                 df_brand_to_export = st.session_state['df_brand_forecast']
            #                 filename_suffix = ""
                        
            #             # Get epidemiology data
            #             if 'df_epi_with_events' in st.session_state:
            #                 df_epi_to_export = st.session_state['df_epi_with_events']
            #             elif 'df_metrics_custom' in st.session_state:
            #                 df_epi_to_export = st.session_state['df_metrics_custom']
            #             else:
            #                 df_epi_to_export = st.session_state['df_metrics_base']
                        
            #             # Create Excel file
            #             output = BytesIO()
            #             with pd.ExcelWriter(output, engine='xlsxwriter') as writer:
            #                 # Write brand forecast
            #                 df_brand_to_export.to_excel(writer, sheet_name='Brand Forecast')
                            
            #                 # Write epidemiology data
            #                 df_epi_to_export.to_excel(writer, sheet_name='Epidemiology')
                            
            #                 # Write brand configuration
            #                 if 'brand_config' in st.session_state:
            #                     config_df = pd.DataFrame([st.session_state['brand_config']]).T
            #                     config_df.columns = ['Value']
            #                     config_df.to_excel(writer, sheet_name='Configuration')
                            
            #                 # Write events if they exist
            #                 if st.session_state.get('brand_events'):
            #                     events_df = pd.DataFrame(st.session_state['brand_events'])
            #                     events_df.to_excel(writer, sheet_name='Brand Events', index=False)
                            
            #                 if st.session_state.get('market_events'):
            #                     market_events_df = pd.DataFrame(st.session_state['market_events'])
            #                     market_events_df.to_excel(writer, sheet_name='Market Events', index=False)
                        
            #             excel_data = output.getvalue()
                        
            #             st.download_button(
            #                 label='📥 Download Excel File',
            #                 data=excel_data,
            #                 file_name=f'{brand_name}_Forecast{filename_suffix}.xlsx',
            #                 mime='application/vnd.openxmlformats-officedocument.spreadsheetml.sheet',
            #                 use_container_width=True
            #             )
                        
            #             st.success('✅ Excel file ready for download!')
                        
            #         except Exception as e:
            #             st.error(f'Error creating Excel file: {str(e)}')
            #             import traceback
            #             st.code(traceback.format_exc())
        
        # ===== COMPLETE FORECAST DISPLAY =====
        # Display complete integrated forecast
        if 'df_brand_forecast' in st.session_state:
            st.markdown("---")
            st.subheader(f'📊 Complete Forecast: Epidemiology + {brand_name}')
            
            # Create tabs for different views
            view_tab1, view_tab2, view_tab3 = st.tabs([
                "Integrated View", 
                "Visualizations", 
                "Separate Tables"
            ])
            
            # ===== TAB 1: INTEGRATED VIEW =====
            with view_tab1:
                st.markdown("### Complete Integrated Forecast Model")
                st.markdown("*This view shows the full patient journey from epidemiology through brand-specific metrics.*")
                
                # Combine epidemiology and brand forecasts
                df_epi_display = df_for_brand.copy()
                df_brand_display = st.session_state['df_brand_forecast'].copy().astype(object)
                
                # **FIX: Ensure both dataframes have the same column type (strings)**
                df_epi_display.columns = df_epi_display.columns.astype(str)
                df_brand_display.columns = df_brand_display.columns.astype(str)
                
                # Get years list (now all strings)
                years = [str(col) for col in df_epi_display.columns if str(col).isdigit()]
                
                # Create a combined display dataframe
                combined_data = {}
                
                # Add all epidemiology rows
                for metric in df_epi_display.index:
                    combined_data[metric] = [df_epi_display.loc[metric, y] for y in years]
                
                # Add separator
                combined_data[''] = [''] * len(years)
                combined_data[f'=== {brand_name.upper()} FORECAST ==='] = [''] * len(years)
                combined_data['  '] = [''] * len(years)
                
                # Add brand configuration info as rows
                brand_config = st.session_state.get('brand_config', {})
                combined_data[f'{brand_name} Launch Year'] = [brand_config.get('launch_year', 'Unknown')] + [''] * (len(years) - 1)
                combined_data[f'Peak Share'] = [f"{brand_config.get('peak_share', 0)*100:.0f}%"] + [''] * (len(years) - 1)
                combined_data[f'Uptake Type'] = [brand_config.get('uptake_type', '7 Year Medium')] + [''] * (len(years) - 1)
                combined_data['   '] = [''] * len(years)
                
                # Add all brand rows
                for metric in df_brand_display.index:
                    combined_data[metric] = [df_brand_display.loc[metric, y] for y in years]
                
                # Create combined dataframe
                df_combined = pd.DataFrame(combined_data, index=years).T
                
                # Format numbers for display
                def format_value(val, row_name):
                    try:
                        if val == '' or pd.isna(val):
                            return ''
                        
                        val_float = float(val)
                        
                        # CRITICAL: Check for COUNT/POPULATION metrics FIRST (before any percentage checks)
                        if ('count' in row_name.lower() or 
                            'population' in row_name.lower() or 
                            'patients' in row_name.lower() or
                            'volume' in row_name.lower() or
                            ('dose' in row_name.lower() and 'price' not in row_name.lower())):
                            return f'{val_float:,.0f}'
                        elif 'rate' in row_name.lower() or 'share' in row_name.lower() or row_name.endswith('%') or row_name.endswith(' %'):
                            return f'{val_float*100:.2f}%'
                        elif 'adherence' in row_name.lower():
                            return f'{val_float*100:.2f}%'
                        elif '$' in row_name or 'price' in row_name.lower():
                            return f'${val_float:,.2f}'
                        elif 'revenue' in row_name.lower() or 'billions' in row_name.lower():
                            return f'${val_float:,.1f}B'
                        else:
                            return f'{val_float:,.0f}'
                    except:
                        return str(val)
                
                # Apply formatting
                df_styled = df_combined.copy().astype(object)
                for col in df_styled.columns:
                    for idx in df_styled.index:
                        df_styled.loc[idx, col] = format_value(df_styled.loc[idx, col], idx)
                
                # Display
                st.dataframe(
                    df_styled,
                    use_container_width=True,
                    height=min(800, len(df_styled) * 35 + 38)
                )
                
                # Add summary metrics
                st.markdown("---")
                st.subheader("📊 Key Performance Indicators")
                
                last_year = years[-1]
                
                kpi_col1, kpi_col2, kpi_col3, kpi_col4 = st.columns(4)
                
                with kpi_col1:
                    total_treated = df_epi_display.loc[target_population, last_year]
                    st.metric(
                        label=f"Total {target_population} ({last_year})",
                        value=f"{float(total_treated):,.0f}"
                    )
                
                with kpi_col2:
                    brand_share = df_brand_display.loc['Brand X Patient Share', last_year]
                    st.metric(
                        label=f"{brand_name} Market Share ({last_year})",
                        value=f"{float(brand_share)*100:.1f}%"
                    )
                
                with kpi_col3:
                    patients_on_brand = df_brand_display.loc['Patients on Brand X', last_year]
                    st.metric(
                        label=f"Patients on {brand_name} ({last_year})",
                        value=f"{float(patients_on_brand):,.0f}"
                    )
                
                with kpi_col4:
                    revenue = df_brand_display.loc['Revenue in billions', last_year]
                    st.metric(
                        label=f"Revenue ({last_year})",
                        value=f"${float(revenue):.1f}B"
                    )
                                 
            # ===== TAB 2: VISUALIZATIONS =====
            with view_tab2:
                st.markdown("### 📈 Visual Analysis")
                
                # Revenue Chart
                st.subheader(f'💰 {brand_name} Revenue Projection')
                revenue_data = df_brand_display.loc['Revenue in billions']
                
                fig_revenue = px.bar(
                    x=revenue_data.index,
                    y=revenue_data.values,
                    labels={'x': 'Year', 'y': 'Revenue ($ Billions)'},
                    title=f'{brand_name} Revenue Forecast',
                    color_discrete_sequence=['#70AD47'],
                    text=revenue_data.values
                )
                fig_revenue.update_traces(texttemplate='$%{text:.1f}B', textposition='outside')
                fig_revenue.update_layout(showlegend=False, height=400)
                st.plotly_chart(fig_revenue, use_container_width=True)
                
                # Patient Volume Charts
                col1, col2 = st.columns(2)
                
                with col1:
                    st.subheader(f'👥 {brand_name} Patient Volume')
                    patient_data = df_brand_display.loc['Patients on Brand X']
                    
                    fig_patients = px.area(
                        x=patient_data.index,
                        y=patient_data.values,
                        labels={'x': 'Year', 'y': 'Patients'},
                        title=f'{brand_name} Patient Volume',
                        color_discrete_sequence=['#4472C4']
                    )
                    fig_patients.update_layout(showlegend=False, height=350)
                    st.plotly_chart(fig_patients, use_container_width=True)
                
                with col2:
                    st.subheader('📊 Market Share Evolution')
                    share_data = df_brand_display.loc['Brand X Patient Share']
                    
                    fig_share = px.line(
                        x=share_data.index,
                        y=[v * 100 for v in share_data.values],
                        labels={'x': 'Year', 'y': 'Market Share (%)'},
                        title=f'{brand_name} Market Share Growth',
                        markers=True,
                        color_discrete_sequence=['#FFA500']
                    )
                    fig_share.update_layout(showlegend=False, height=350)
                    st.plotly_chart(fig_share, use_container_width=True)
                
                # Patient Funnel Chart
                st.subheader('🔄 Patient Flow Analysis (Latest Year)')
                last_year = years[-1]
                
                # Get funnel data
                funnel_data = []
                
                # Find relevant metrics
                if target_population in df_epi_display.index:
                    target_pop_value = df_epi_display.loc[target_population, last_year]
                    funnel_data.append({'Stage': target_population, 'Patients': target_pop_value})
                
                if 'Patients on Brand X' in df_brand_display.index:
                    brand_patients = df_brand_display.loc['Patients on Brand X', last_year]
                    funnel_data.append({'Stage': f'{brand_name} Patients', 'Patients': brand_patients})
                
                if 'Adherent Patients' in df_brand_display.index:
                    adherent = df_brand_display.loc['Adherent Patients', last_year]
                    funnel_data.append({'Stage': 'Adherent Patients', 'Patients': adherent})
                
                if funnel_data:
                    df_funnel = pd.DataFrame(funnel_data)
                    fig_funnel = px.funnel(
                        df_funnel,
                        x='Patients',
                        y='Stage',
                        title=f'Patient Funnel - {last_year}',
                        color_discrete_sequence=['#70AD47']
                    )
                    fig_funnel.update_layout(height=300)
                    st.plotly_chart(fig_funnel, use_container_width=True)
            
            # ===== TAB 3: SEPARATE TABLES =====
            with view_tab3:
                st.markdown("### 📋 Detailed Data Tables")
                
                # Epidemiology Forecast
                st.subheader('🔬 Epidemiology Forecast')
                df_epi_formatted = df_epi_display.copy().astype(object)
                
                # Format epidemiology data
                for idx in df_epi_formatted.index:
                    if 'rate' in idx.lower():
                        df_epi_formatted.loc[idx] = df_epi_formatted.loc[idx].apply(lambda x: f'{float(x)*100:.2f}%')
                    else:
                        df_epi_formatted.loc[idx] = df_epi_formatted.loc[idx].apply(lambda x: f'{float(x):,.0f}')
                
                st.dataframe(df_epi_formatted, use_container_width=True)
                
                st.markdown("---")
                
                # Brand Forecast
                st.subheader(f'💊 {brand_name} Forecast')
                
                # Show configuration
                config_data = {
                    'Parameter': [
                        'Launch Year',
                        'Peak Market Share',
                        'Uptake Curve',
                        'Target Population',
                        'Adherence Rate',
                        'Annual Doses per Patient',
                        'Price per Dose'
                    ],
                    'Value': [
                        st.session_state['brand_config'].get('launch_year', 'Unknown'),
                        f"{st.session_state['brand_config'].get('peak_share', 0)*100:.1f}%",
                        st.session_state['brand_config'].get('uptake_type', '7 Year Medium'),
                        st.session_state['brand_config'].get('target_population', 'Unknown'),
                        f"{st.session_state['brand_config'].get('adherence', 0)*100:.1f}%",
                        st.session_state['brand_config'].get('annual_dose', 0),
                        f"${st.session_state['brand_config'].get('price_per_dose', 0):.2f}"
                    ]
                }
                st.table(pd.DataFrame(config_data))
                
                st.markdown("---")
                
                # Format brand data
                df_brand_formatted = df_brand_display.copy()

                for idx in df_brand_formatted.index:
                    if 'volume' in idx.lower() or ('dose' in idx.lower() and 'price' not in idx.lower()):
                        df_brand_formatted.loc[idx] = df_brand_formatted.loc[idx].apply(lambda x: f'{float(x):,.0f}')
                    elif 'share' in idx.lower() or ('rate' in idx.lower() and 'volume' not in idx.lower()) or ('adherence' in idx.lower() and 'volume' not in idx.lower()):
                        df_brand_formatted.loc[idx] = df_brand_formatted.loc[idx].apply(lambda x: f'{float(x)*100:.2f}%')
                    elif 'revenue' in idx.lower() or 'billions' in idx.lower():
                        df_brand_formatted.loc[idx] = df_brand_formatted.loc[idx].apply(lambda x: f'${float(x):,.1f}B')
                    elif 'price' in idx.lower() or '$' in idx:
                        df_brand_formatted.loc[idx] = df_brand_formatted.loc[idx].apply(lambda x: f'${float(x):,.2f}')
                    else:
                        df_brand_formatted.loc[idx] = df_brand_formatted.loc[idx].apply(lambda x: f'{float(x):,.0f}')                
                st.dataframe(df_brand_formatted, use_container_width=True)
            
            # Download button for brand forecast
            st.markdown("---")
            st.subheader('📥 Export Complete Model')
            
            col1, col2 = st.columns([3, 1])
            
            with col1:
                st.info(f"""
                **Your complete forecast is ready for download!**
                
                The Excel file includes:
                - ✅ Full Epidemiology Forecast (with formulas)
                - ✅ Brand Forecast with dynamic calculations
                - ✅ Uptake curves reference sheet
                - ✅ Research citations
                - ✅ Comprehensive instructions
                
                All calculations are formula-based so you can modify parameters and see instant results!
                """)
            
            with col2:
                excel_name_brand = f"complete_forecast_{brand_name.replace(' ', '_')}_{datetime.now().strftime('%Y%m%d_%H%M%S')}.xlsx"
                excel_bytes_brand = export_brand_to_excel(
                    df_for_brand,
                    st.session_state['df_brand_forecast'],
                    st.session_state['brand_config'],
                    st.session_state.get('ai_assumptions_v2', {}),
                    excel_name_brand
                )
                st.download_button(
                    '📥 Download Complete Excel Model',
                    data=excel_bytes_brand,
                    file_name=excel_name_brand,
                    mime='application/vnd.ms-excel',
                    type='primary',
                    use_container_width=True
                )
    
    # ============================================================
    # TAB 7: SCENARIO MANAGEMENT
    # ============================================================
    with tab_scenarios:
        st.header('🔄 Scenario Management')
        
        st.markdown("""
        Create and compare multiple forecasting scenarios with different assumptions.
        
        **Workflow:**
        1. Generate a complete forecast (epidemiology + brand)
        2. Save it as a scenario
        3. Modify parameters (e.g., change peak share, price, uptake)
        4. Generate and save as a new scenario
        5. Compare scenarios side-by-side
        
        Save this forecast as a scenario, then modify parameters to create alternative scenarios and compare them side-by-side.
        """)
        
        # Initialize saved scenarios
        if 'saved_scenarios' not in st.session_state:
            st.session_state['saved_scenarios'] = {}
        
        # Initialize events in session state
        if 'market_events' not in st.session_state:
            st.session_state['market_events'] = []
        if 'brand_events' not in st.session_state:
            st.session_state['brand_events'] = []
        
        # Event type selector
        st.markdown("### Event Management")
        event_scope = st.radio(
            "📍 Event Scope",
            options=['Brand Event (Market Share)', 'Market Event (Epidemiology)'],
            horizontal=True,
            help="Choose whether this event affects your brand or the base epidemiology forecast"
        )
        
        is_market_event = (event_scope == 'Market Event (Epidemiology)')
        
        # Create tabs for event workflow
        if is_market_event:
            event_tab1, event_tab2, event_tab3 = st.tabs([
                "➕ Add Market Event",
                "📋 Manage Market Events",
                "🔄 Apply Events & Regenerate"
            ])
        else:
            event_tab1, event_tab2, event_tab3 = st.tabs([
                "➕ Add Brand Event",
                "📋 Manage Brand Events",
                "🔄 Apply Events & Regenerate"
            ])
        
        # ===== TAB 1: ADD EVENT =====
        with event_tab1:
            if is_market_event:
                st.subheader("➕ Add Market Event")
                
                st.info("🌍 Market Events affect the epidemiology forecast BEFORE brand calculations. Examples: Regulatory changes, new treatment guidelines, screening program changes.")
                
                # Get available metrics from forecast
                if 'df_metrics_custom' in st.session_state:
                    df_for_events = st.session_state['df_metrics_custom']
                else:
                    df_for_events = st.session_state.get('df_metrics_base')
                
                if df_for_events is not None:
                    available_metrics = [m for m in df_for_events.index 
                                       if m.strip() and not m.startswith('===') and not m.startswith('  ')]
                    
                    years = [int(col) for col in df_for_events.columns if str(col).isdigit()]
                    min_year = min(years) if years else 2025
                    max_year = max(years) if years else 2035
                    
                    col1, col2 = st.columns(2)
                    
                    with col1:
                        market_event_type = st.selectbox(
                            "📌 Event Type",
                            options=[
                                'Regulatory Change',
                                'New Treatment Guidelines',
                                'Screening Program Change',
                                'Healthcare Policy Change',
                                'Market Access Restriction',
                                'Reimbursement Change',
                                'Custom Market Event'
                            ],
                            key='market_event_type_select'
                        )
                        
                        market_event_year = st.number_input(
                            "📆 Event Year",
                            min_value=min_year,
                            max_value=max_year,
                            value=min_year + 2,
                            key='market_event_year_input'
                        )
                        
                        market_event_description = st.text_area(
                            "📝 Event Description",
                            placeholder="e.g., 'New regulations restrict treatment to severe cases only'",
                            key='market_event_desc_input',
                            height=80
                        )
                    
                    with col2:
                        st.markdown("**Impact Configuration**")
                        
                        target_metric = st.selectbox(
                            "🎯 Target Metric",
                            options=available_metrics,
                            key='market_target_metric',
                            help="Which metric will be affected by this event?"
                        )
                        
                        impact_type = st.radio(
                            "Impact Type",
                            options=['Percentage Change', 'Absolute Change'],
                            key='market_impact_type',
                            help="How to apply the change"
                        )
                        
                        if impact_type == 'Percentage Change':
                            impact_value = st.number_input(
                                "Impact (% change)",
                                min_value=-100.0,
                                max_value=100.0,
                                value=-10.0,
                                step=1.0,
                                key='market_impact_pct',
                                help="Negative for decrease, positive for increase"
                            )
                            impact_value_normalized = impact_value / 100.0
                            impact_display = f"{impact_value:+.1f}%"
                        else:
                            current_value = df_for_events.loc[target_metric, str(years[0])]
                            st.info(f"Current value (first year): {current_value:,.0f}")
                            
                            impact_value = st.number_input(
                                "Impact (absolute change)",
                                value=-1000.0,
                                step=100.0,
                                key='market_impact_abs',
                                help="Absolute number to add/subtract"
                            )
                            impact_value_normalized = impact_value
                            impact_display = f"{impact_value:+,.0f}"
                    
                    # Add button
                    st.markdown("---")
                    if st.button("➕ Add Market Event", type="primary", use_container_width=True):
                        new_event = {
                            'event_year': market_event_year,
                            'event_type': market_event_type,
                            'description': market_event_description,
                            'target_metric': target_metric,
                            'impact_type': 'percentage_change' if impact_type == 'Percentage Change' else 'absolute_change',
                            'impact_value': impact_value_normalized
                        }
                        
                        st.session_state['market_events'].append(new_event)
                        st.success(f"✅ Market event added: {market_event_type} affecting {target_metric}")
                        st.info("Go to 'Apply Events & Regenerate' tab to see the updated forecast")
                        st.rerun()
                else:
                    st.warning("Generate an epidemiology forecast first to add market events.")
            
            else:
                # BRAND EVENT
                st.subheader("➕ Add Brand Event")
                
                st.info("💊 Brand Events affect your brand's market share and dynamics. Examples: Loss of Exclusivity, biosimilar entry, new competitor launches, label expansions.")
                
                # Add tabs for different input methods
                add_method_tab1, add_method_tab2 = st.tabs([
                    "📰 From News Discovery",
                    "✏️ Manual Entry"
                ])
                
                # ===== FROM NEWS DISCOVERY =====
                with add_method_tab1:
                    st.markdown("### 📰 Discover Events from News")
                    
                    # Check if news module is available
                    news_available = NEWS_MODULE_AVAILABLE
                    
                    if news_available:
                        brand_name_for_news = st.session_state.get('brand_config', {}).get('brand_name', 'Brand X')
                        disease_name_for_news = st.session_state.get('disease_name', '')
                        
                        col_search1, col_search2 = st.columns([3, 1])
                        with col_search1:
                            news_search_query = st.text_input(
                                "🔍 Search Query",
                                value=f"{brand_name_for_news} {disease_name_for_news}",
                                help="Enter search terms to find relevant pharmaceutical news",
                                key='news_search_query_input'
                            )
                        with col_search2:
                            st.markdown("<br>", unsafe_allow_html=True)
                            search_news_btn = st.button("🔎 Search News", use_container_width=True, key='search_news_button')
                        
                        if search_news_btn and news_search_query:
                            with st.spinner("Searching pharmaceutical news..."):
                                try:
                                    news_results = fetch_drug_news(news_search_query, n_results=20)
                                    st.session_state['news_results'] = news_results
                                    st.session_state['news_display_count'] = 5  # Initially show 5
                                    st.success(f"✅ Found {len(news_results)} news articles")
                                except Exception as e:
                                    st.error(f"Error fetching news: {str(e)}")
                        
                        # Display news results and allow selection
                        if 'news_results' in st.session_state and st.session_state['news_results']:
                            st.markdown("---")
                            st.markdown("#### 📋 Select News to Create Event")
                            
                            # Initialize display count if not exists
                            if 'news_display_count' not in st.session_state:
                                st.session_state['news_display_count'] = 5
                            
                            total_news = len(st.session_state['news_results'])
                            display_count = min(st.session_state['news_display_count'], total_news)
                            
                            # Display news items
                            for idx in range(display_count):
                                news_item = st.session_state['news_results'][idx]
                                with st.expander(f"📰 {news_item.get('title', 'No title')}", expanded=False):
                                    st.markdown(f"**Source:** {news_item.get('source', 'Unknown')}")
                                    st.markdown(f"**Date:** {news_item.get('date', 'Unknown')}")
                                    if news_item.get('summary'):
                                        st.markdown(f"**Summary:** {news_item.get('summary')}")
                                    
                                    st.markdown("---")
                                    st.markdown("**🎯 Event Configuration**")
                                    
                                    col_news1, col_news2 = st.columns(2)
                                    
                                    with col_news1:
                                        event_year_news = st.number_input(
                                            "📆 Event Year",
                                            min_value=2025,
                                            max_value=2040,
                                            value=2026,
                                            key=f'news_event_year_{idx}'
                                        )
                                        
                                        post_share_news = st.number_input(
                                            "📊 New Peak Share (%)",
                                            min_value=0.0,
                                            max_value=100.0,
                                            value=10.0,
                                            step=0.5,
                                            key=f'news_event_share_{idx}',
                                            help="Market share after this event"
                                        )
                                    
                                    with col_news2:
                                        years_to_peak_news = st.number_input(
                                            "⏱️ Time to Peak (Years)",
                                            min_value=1,
                                            max_value=10,
                                            value=5,
                                            key=f'news_years_to_peak_{idx}'
                                        )
                                        
                                        curve_type_news = st.selectbox(
                                            "📈 Uptake/Erosion Curve",
                                            options=['5 Year Medium', '5 Year Fast', '5 Year Slow', 
                                                    '7 Year Medium', '3 Year Fast'],
                                            key=f'news_curve_type_{idx}'
                                        )
                                    
                                    if st.button(f"➕ Add Event from News", key=f'add_news_event_{idx}', type='primary', use_container_width=True):
                                        new_event = {
                                            'event_year': event_year_news,
                                            'event_type': 'custom_event',
                                            'post_event_peak_share': post_share_news / 100.0,
                                            'years_to_peak': years_to_peak_news,
                                            'curve_type': curve_type_news,
                                            'curve_category': 'Recovery (Growth)',
                                            'description': f"📰 {news_item.get('title', 'News Event')}"
                                        }
                                        
                                        if 'brand_events' not in st.session_state:
                                            st.session_state['brand_events'] = []
                                        
                                        st.session_state['brand_events'].append(new_event)
                                        st.success(f"✅ Event added from news: {news_item.get('title', 'Event')[:50]}...")
                                        st.info("Go to 'Manage Brand Events' tab to view or 'Apply Events & Regenerate' to see impact")
                                        # Don't rerun - just show success message
                            
                            # Show "Load More" button if there are more news items
                            if display_count < total_news:
                                st.markdown("---")
                                col_load1, col_load2, col_load3 = st.columns([1, 2, 1])
                                with col_load2:
                                    if st.button(f"📰 Load More News ({display_count}/{total_news})", use_container_width=True, key='load_more_news'):
                                        st.session_state['news_display_count'] = min(display_count + 5, total_news)
                                        st.rerun()
                        else:
                            st.info("👆 Click 'Search News' to discover relevant pharmaceutical news and create events")
                    else:
                        st.warning("📰 News module not available. News discovery feature is disabled.")
                        st.info("To enable news discovery, ensure `news_integrated.py` is available in your workspace.")
                
                # ===== MANUAL ENTRY =====
                with add_method_tab2:
                    # Create sub-tabs for manual entry and CSV upload
                    manual_subtab1, manual_subtab2 = st.tabs([
                        "✏️ Manual Entry",
                        "📄 Upload CSV"
                    ])
                    
                    # ===== MANUAL ENTRY SUB-TAB =====
                    with manual_subtab1:
                        st.markdown("### ✏️ Add Brand Event Manually")
                        
                        col1, col2 = st.columns(2)
                    
                        col1, col2 = st.columns(2)
                        
                        with col1:
                            event_type = st.selectbox(
                                '📅 Event Type',
                                ['loss_of_exclusivity', 'competitor_entry', 'indication_expansion', 'regulatory_change', 'custom_event'],
                                format_func=lambda x: x.replace('_', ' ').title(),
                                key='new_event_type'
                            )
                            event_year = st.number_input(
                                '📆 Event Year',
                                min_value=2025,
                                max_value=2040,
                                value=2030,
                                key='new_event_year'
                            )
                        
                        with col2:
                            post_event_share = st.number_input(
                                '📊 Post-Event Peak Share (%)',
                                min_value=0.0,
                                max_value=100.0,
                                value=15.0,
                                step=0.1,
                                key='new_event_share',
                                help="Market share after the event occurs"
                            )
                            
                            years_to_peak = st.number_input(
                                '⏱️ Years to Reach Post-Event Peak',
                                min_value=1,
                                max_value=8,
                                value=3,
                                key='new_event_years'
                            )
                        
                        event_description = st.text_area(
                            '📝 Event Description',
                            placeholder='Describe the market event and its expected impact...',
                            key='new_event_description'
                        )
                        
                        # Curve selection for post-event behavior
                        if event_type == 'loss_of_exclusivity':
                            st.markdown("**📉 Post-LoE Behavior (Erosion Curve):**")
                            curve_category = 'Erosion (Decline)'
                            curve_options = [k for k in EROSION_CURVES.keys()]
                            default_curve = f"{years_to_peak} Year Erosion Medium"
                        else:
                            st.markdown("**📈 Post-Event Growth (Recovery Curve):**")
                            curve_category = 'Recovery (Growth)'
                            curve_options = [k for k in UPTAKE_CURVES.keys()]
                            default_curve = f"{years_to_peak} Year Medium"
                        
                        if default_curve not in curve_options and curve_options:
                            default_curve = curve_options[0]
                        
                        curve_type = st.selectbox(
                            '📈 Market Response Curve',
                            curve_options,
                            index=curve_options.index(default_curve) if default_curve in curve_options else 0,
                            key='new_event_curve'
                        )
                        
                        if st.button('📅 Add Event', type='primary', use_container_width=True):
                            new_event = {
                                'event_year': event_year,
                                'event_type': event_type,
                                'post_event_peak_share': post_event_share / 100.0,
                                'years_to_peak': years_to_peak,
                                'curve_type': curve_type,
                                'curve_category': curve_category,
                                'description': event_description
                            }
                            
                            if 'brand_events' not in st.session_state:
                                st.session_state['brand_events'] = []
                            
                            st.session_state['brand_events'].append(new_event)
                            st.success(f'✅ {event_type.replace("_", " ").title()} event added for {event_year}')
                            st.rerun()
                        
                        # Note about regenerating forecast
                        if st.session_state.get('brand_events'):
                            st.info("💡 Remember to regenerate the brand forecast after adding/removing events to see the updated results.")
                    
                    # ===== CSV UPLOAD SUB-TAB =====
                    with manual_subtab2:
                        st.markdown("### 📄 Upload Events from CSV")
                        
                        st.markdown("""
                        Upload a CSV file containing multiple brand events. The CSV should have these columns:
                        
                        | Column | Type | Description | Example |
                        |--------|------|-------------|---------|
                        | `event_year` | Integer | Year of event | 2027 |
                        | `event_type` | String | Event category | competitor_entry |
                        | `event_name` | String | Event name | Biosimilar A |
                        | `event_description` | String | Details | First biosimilar competitor |
                        | `post_event_peak_share` | Decimal | Target share (0-1) | 0.12 |
                        | `years_to_peak` | Integer | Years to reach target | 3 |
                        | `curve_type` | String | Curve name | 3 Year Medium Erosion |
                        | `curve_category` | String | Growth or decline | Erosion (Decline) |
                        
                        **Download Template:** [brand_events.csv](brand_events.csv)
                        """)
                        
                        # File uploader
                        uploaded_csv = st.file_uploader(
                            "📤 Upload CSV File",
                            type=['csv'],
                            key='brand_events_csv_upload',
                            help="Upload a CSV file with brand events"
                        )
                        
                        if uploaded_csv is not None:
                            try:
                                # Read CSV

                                df_events = pd.read_csv(uploaded_csv)
                                
                                # Validate required columns
                                required_cols = ['event_year', 'event_type', 'post_event_peak_share', 
                                               'years_to_peak', 'curve_type', 'curve_category']
                                missing_cols = [col for col in required_cols if col not in df_events.columns]
                                
                                if missing_cols:
                                    st.error(f"❌ Missing required columns: {', '.join(missing_cols)}")
                                    st.info("Required columns: " + ", ".join(required_cols))
                                else:
                                    # Display preview
                                    st.success(f"✅ CSV loaded successfully! Found {len(df_events)} events")
                                    
                                    with st.expander("📋 Preview Events", expanded=True):
                                        st.dataframe(df_events, use_container_width=True)
                                    
                                    # Validate data
                                    validation_errors = []
                                    
                                    for idx, row in df_events.iterrows():
                                        # Validate event_year is numeric
                                        try:
                                            year = int(row['event_year'])
                                            if year < 2020 or year > 2050:
                                                validation_errors.append(f"Row {idx+1}: event_year {year} out of range (2020-2050)")
                                        except:
                                            validation_errors.append(f"Row {idx+1}: event_year must be a valid integer")
                                        
                                        # Validate post_event_peak_share
                                        try:
                                            share = float(row['post_event_peak_share'])
                                            if share < 0 or share > 1:
                                                validation_errors.append(f"Row {idx+1}: post_event_peak_share must be between 0 and 1 (found {share})")
                                        except:
                                            validation_errors.append(f"Row {idx+1}: post_event_peak_share must be a valid decimal")
                                        
                                        # Validate years_to_peak
                                        try:
                                            ytp = int(row['years_to_peak'])
                                            if ytp < 1 or ytp > 10:
                                                validation_errors.append(f"Row {idx+1}: years_to_peak must be between 1 and 10")
                                        except:
                                            validation_errors.append(f"Row {idx+1}: years_to_peak must be a valid integer")
                                    
                                    if validation_errors:
                                        st.error("⚠️ Validation Errors Found:")
                                        for error in validation_errors:
                                            st.error(f"  • {error}")
                                    else:
                                        st.success("✅ All validations passed!")
                                        
                                        # Add button
                                        col_csv1, col_csv2 = st.columns([3, 1])
                                        with col_csv2:
                                            if st.button("➕ Add All Events", type="primary", use_container_width=True, key='add_csv_events_btn'):
                                                # Convert DataFrame to list of event dictionaries
                                                new_events = []
                                                
                                                for idx, row in df_events.iterrows():
                                                    event = {
                                                        'event_year': int(row['event_year']),
                                                        'event_type': str(row['event_type']),
                                                        'event_name': str(row.get('event_name', f"Event {idx+1}")),
                                                        'post_event_peak_share': float(row['post_event_peak_share']),
                                                        'years_to_peak': int(row['years_to_peak']),
                                                        'curve_type': str(row['curve_type']),
                                                        'curve_category': str(row['curve_category']),
                                                        'description': str(row.get('event_description', row.get('event_name', f"Event {idx+1}")))
                                                    }
                                                    new_events.append(event)
                                                
                                                # Sort events by event_year (CRITICAL for proper impact calculation)
                                                new_events.sort(key=lambda x: x['event_year'])
                                                
                                                # Initialize brand_events if needed
                                                if 'brand_events' not in st.session_state:
                                                    st.session_state['brand_events'] = []
                                                
                                                # Add events
                                                st.session_state['brand_events'].extend(new_events)
                                                
                                                # Sort all events by year
                                                st.session_state['brand_events'].sort(key=lambda x: x['event_year'])
                                                
                                                st.success(f"✅ Successfully added {len(new_events)} events (sorted by year)")
                                                st.info("📊 Go to 'Manage Brand Events' tab to review or 'Apply Events & Regenerate' to see impact")
                                                
                                                # Show summary
                                                st.markdown("**Events Added:**")
                                                for event in new_events:
                                                    st.write(f"  • {event['event_year']}: {event.get('event_name', event['event_type'])} → {event['post_event_peak_share']*100:.1f}% share")
                                                
                                                
                                                
                            except Exception as e:
                                st.error(f"❌ Error reading CSV: {str(e)}")
                                st.info("Please ensure your CSV follows the correct format.")
                        
                        else:
                            st.info("👆 Upload a CSV file to bulk-add brand events")

        
        # ===== TAB 2: MANAGE EVENTS =====
        with event_tab2:
            if is_market_event:
                st.subheader("📋 Active Market Events")
                
                if not st.session_state['market_events']:
                    st.info("No market events added yet.")
                else:
                    for idx, event in enumerate(st.session_state['market_events']):
                        with st.expander(f"⚡ {event['event_type']} - Year {event['event_year']}", expanded=False):
                            col1, col2, col3 = st.columns([2, 2, 1])
                            
                            with col1:
                                st.markdown("**Event Details**")
                                st.write(f"**Year:** {event['event_year']}")
                                st.write(f"**Type:** {event['event_type']}")
                                if event.get('description'):
                                    st.write(f"**Description:** {event['description']}")
                            
                            with col2:
                                st.markdown("**Impact**")
                                st.write(f"**Target:** {event['target_metric']}")
                                impact_display = f"{event['impact_value']*100:+.1f}%" if event['impact_type'] == 'percentage_change' else f"{event['impact_value']:+,.0f}"
                                st.write(f"**Change:** {impact_display}")
                            
                            with col3:
                                if st.button("🗑️", key=f"remove_market_{idx}", use_container_width=True):
                                    st.session_state['market_events'].pop(idx)
                                    st.success("Event removed")
                                    st.rerun()
                    
                    if st.button("🗑️ Clear All Market Events", type="secondary"):
                        st.session_state['market_events'] = []
                        st.rerun()
            
            else:
                st.subheader("📋 Active Brand Events")
                
                if not st.session_state['brand_events']:
                    st.info("No brand events added yet.")
                else:
                    for idx, event in enumerate(st.session_state['brand_events']):
                        with st.expander(f"💊 {event.get('event_type', 'Event')} - Year {event['event_year']}", expanded=False):
                            col1, col2, col3 = st.columns([2, 2, 1])
                            
                            with col1:
                                st.markdown("**Event Details**")
                                st.write(f"**Year:** {event['event_year']}")
                                st.write(f"**Type:** {event.get('event_type', 'N/A')}")
                                if event.get('description'):
                                    st.write(f"**Description:** {event['description']}")
                            
                            with col2:
                                st.markdown("**Impact**")
                                st.write(f"**Post-Event Share:** {event['post_event_peak_share']*100:.1f}%")
                                st.write(f"**Curve:** {event.get('curve_type', 'N/A')}")
                                st.write(f"**Category:** {event.get('curve_category', 'N/A')}")
                            
                            with col3:
                                if st.button("🗑️", key=f"remove_brand_{idx}", use_container_width=True):
                                    st.session_state['brand_events'].pop(idx)
                                    st.success("Event removed")
                                    st.rerun()
                    
                    if st.button("🗑️ Clear All Brand Events", type="secondary"):
                        st.session_state['brand_events'] = []
                        st.rerun()
        
        # ===== TAB 3: APPLY AND REGENERATE =====
        with event_tab3:
            st.subheader("🔄 Generate Forecast with Events")
            
            has_market_events = len(st.session_state.get('market_events', [])) > 0
            has_brand_events = len(st.session_state.get('brand_events', [])) > 0
            
            if not has_market_events and not has_brand_events:
                st.warning("No events to apply. Add market or brand events first.")
            else:
                # Show summary
                col1, col2 = st.columns(2)
                
                with col1:
                    st.markdown("### 🌍 Market Events")
                    if has_market_events:
                        st.success(f"{len(st.session_state['market_events'])} market event(s) ready")
                        for event in st.session_state['market_events']:
                            st.caption(f"• {event['event_year']}: {event['event_type']}")
                    else:
                        st.info("No market events")
                
                with col2:
                    st.markdown("### 💊 Brand Events")
                    if has_brand_events:
                        st.success(f"{len(st.session_state['brand_events'])} brand event(s) ready")
                        for event in st.session_state['brand_events']:
                            st.caption(f"• {event['event_year']}: {event.get('event_type', 'Event')}")
                    else:
                        st.info("No brand events")
                
                st.markdown("---")
                
                if st.button("📊 Generate Forecast with Events", type="primary", use_container_width=True):
                    with st.spinner("Generating forecast with events..."):
                        try:
                            # Step 1: Get base forecast
                            if 'df_metrics_custom' in st.session_state:
                                df_base = st.session_state['df_metrics_custom']
                            else:
                                df_base = st.session_state['df_metrics_base']
                            
                            years = [int(col) for col in df_base.columns if str(col).isdigit()]
                            start_year_val = min(years) if years else 2025
                            end_year_val = max(years) if years else 2030
                            
                            # Step 2: Apply market events if any
                            if has_market_events:
                                df_epi_with_events = apply_market_events(
                                    df_base,
                                    st.session_state['market_events'],
                                    start_year_val
                                )
                            else:
                                df_epi_with_events = df_base.copy()
                            
                            # Step 3: Generate brand forecast with events
                            if 'brand_config' in st.session_state:
                                df_brand_with_events = generate_brand_forecast(
                                    df_epi_with_events,
                                    st.session_state['brand_config'],
                                    start_year_val,
                                    end_year_val,
                                    events=st.session_state['brand_events'] if has_brand_events else None
                                )
                                
                                # Store the forecasts with events
                                st.session_state['df_epi_with_events'] = df_epi_with_events
                                st.session_state['df_brand_with_events'] = df_brand_with_events
                                
                                st.success("✅ Forecast with events generated successfully!")
                                st.info("📊 Download the updated forecast below or view in Brand Analysis tab")
                            else:
                                st.error("Please generate a brand forecast first in the Brand Analysis tab.")
                            
                        except Exception as e:
                            st.error(f"Error: {str(e)}")
                            import traceback
                            st.code(traceback.format_exc())
                
                # ===== DISPLAY BOTH FORECASTS SEPARATELY =====
                if 'df_brand_with_events' in st.session_state:
                    st.markdown("---")
                    st.markdown("---")
                    st.markdown('<div class="section-header"><span class="section-icon">📊</span><h2>Forecast Comparison</h2></div>', unsafe_allow_html=True)
                    
                    # Get both forecasts
                    if 'df_metrics_custom' in st.session_state:
                        df_epi_original = st.session_state['df_metrics_custom']
                    else:
                        df_epi_original = st.session_state['df_metrics_base']
                    
                    df_epi_with_events = st.session_state.get('df_epi_with_events', df_epi_original)
                    df_brand_original = st.session_state['df_brand_forecast']
                    df_brand_with_events = st.session_state['df_brand_with_events']
                    
                    # Create comparison tabs
                    compare_tab1, compare_tab2, compare_tab3 = st.tabs([
                        "📊 Original Forecast",
                        "⚡ Forecast with Events",
                        "📥 Download with Events"
                    ])
                    
                    # ===== TAB 1: ORIGINAL FORECAST =====
                    with compare_tab1:
                        st.markdown("### 📊 Original Forecast (Without Events)")
                        
                        st.markdown("#### 🔬 Epidemiology")
                        df_epi_display = df_epi_original.copy()
                        
                        # Format epidemiology data
                        for idx in df_epi_display.index:
                            if 'rate' in idx.lower() or '%' in idx:
                                df_epi_display.loc[idx] = df_epi_display.loc[idx].apply(lambda x: f'{float(x)*100:.2f}%' if pd.notna(x) and x != '' else '')
                            else:
                                df_epi_display.loc[idx] = df_epi_display.loc[idx].apply(lambda x: f'{float(x):,.0f}' if pd.notna(x) and x != '' else '')
                        
                        st.dataframe(df_epi_display, use_container_width=True)
                        
                        st.markdown("---")
                        st.markdown(f"#### 💊 {brand_name} Forecast")
                        df_brand_display = df_brand_original.copy()
                        
                        # Format brand data with proper volume/dose handling
                        for idx in df_brand_display.index:
                            # Check for volume/dose FIRST (before percentage checks)
                            if 'volume' in idx.lower() or ('dose' in idx.lower() and 'price' not in idx.lower()):
                                df_brand_display.loc[idx] = df_brand_display.loc[idx].apply(lambda x: f'{float(x):,.0f}' if pd.notna(x) and x != '' else '')
                            # Then check for rates/shares (but NOT if it's a volume/dose metric)
                            elif 'share' in idx.lower() or ('rate' in idx.lower() and 'volume' not in idx.lower()):
                                df_brand_display.loc[idx] = df_brand_display.loc[idx].apply(lambda x: f'{float(x)*100:.2f}%' if pd.notna(x) and x != '' else '')
                            # Adherence as percentage (but already caught volume above)
                            elif 'adherence' in idx.lower() and 'volume' not in idx.lower():
                                df_brand_display.loc[idx] = df_brand_display.loc[idx].apply(lambda x: f'{float(x)*100:.2f}%' if pd.notna(x) and x != '' else '')
                            # Revenue
                            elif 'revenue' in idx.lower() or 'billions' in idx.lower():
                                df_brand_display.loc[idx] = df_brand_display.loc[idx].apply(lambda x: f'${float(x):,.1f}B' if pd.notna(x) and x != '' else '')
                            # Price
                            elif 'price' in idx.lower() or '$' in idx:
                                df_brand_display.loc[idx] = df_brand_display.loc[idx].apply(lambda x: f'${float(x):,.2f}' if pd.notna(x) and x != '' else '')
                            # Default: whole numbers for patient counts and volumes
                            else:
                                df_brand_display.loc[idx] = df_brand_display.loc[idx].apply(lambda x: f'{float(x):,.0f}' if pd.notna(x) and x != '' else '')
                        
                        st.dataframe(df_brand_display, use_container_width=True)
                    
                    # ===== TAB 2: FORECAST WITH EVENTS =====
                    with compare_tab2:
                        st.markdown("### ⚡ Forecast with Events Applied")
                        
                        # Show which events are applied
                        col1, col2 = st.columns(2)
                        
                        with col1:
                            if has_market_events:
                                st.markdown("**🌍 Applied Market Events:**")
                                for event in st.session_state['market_events']:
                                    impact_display = f"{event['impact_value']*100:+.1f}%" if event['impact_type'] == 'percentage_change' else f"{event['impact_value']:+,.0f}"
                                    st.info(f"• **{event['event_year']}**: {event['event_type']}\n- Target: {event['target_metric']}\n- Impact: {impact_display}")
                        
                        with col2:
                            if has_brand_events:
                                st.markdown("**💊 Applied Brand Events:**")
                                for event in st.session_state['brand_events']:
                                    st.info(f"• **{event['event_year']}**: {event.get('event_type', 'Event')}\n- New Peak: {event['post_event_peak_share']*100:.1f}%\n- Curve: {event.get('curve_type', 'N/A')}")
                        
                        st.markdown("---")
                        
                        # ===== GRAPHICAL COMPARISONS =====
                        st.markdown("### 📊 Visual Impact Analysis")
                        
                        # Get year columns from brand forecast (not epi forecast)
                        year_cols = [col for col in df_brand_original.columns if str(col).isdigit()]
                        
                        # If no numeric columns found, try converting column names
                        if not year_cols:
                            year_cols = [col for col in df_brand_original.columns if isinstance(col, (int, str)) and str(col).replace('.', '').isdigit()]
                        
                        # Get brand name from config
                        brand_name_clean = brand_name.replace('Brand ', '').strip()
                        
                        # 1. Revenue Comparison (CRITICAL METRIC)
                        revenue_metric_found = False
                        for possible_name in ['Revenue in billions', f'{brand_name} Revenue (Billions $)', 'Revenue (Billions $)']:
                            if possible_name in df_brand_original.index and len(year_cols) > 0:
                                try:
                                    revenue_original = df_brand_original.loc[possible_name, year_cols].astype(float)
                                    revenue_events = df_brand_with_events.loc[possible_name, year_cols].astype(float)
                                    revenue_metric_found = True
                                    
                                    # Bar Chart
                                    fig_revenue = go.Figure()
                                    fig_revenue.add_trace(go.Bar(
                                        x=year_cols,
                                        y=revenue_original,
                                        name='Without Events',
                                        marker_color='#93C5FD',
                                        opacity=0.7
                                    ))
                                    fig_revenue.add_trace(go.Bar(
                                        x=year_cols,
                                        y=revenue_events,
                                        name='With Events',
                                        marker_color='#34D399',
                                        opacity=0.7
                                    ))
                                    fig_revenue.update_layout(
                                        title=f'{brand_name} Revenue: Impact of Events',
                                        xaxis_title='Year',
                                        yaxis_title='Revenue (Billions $)',
                                        barmode='group',
                                        hovermode='x unified',
                                        height=400
                                    )
                                    st.plotly_chart(fig_revenue, use_container_width=True)
                                    break
                                except Exception as e:
                                    st.error(f"Error plotting revenue for {possible_name}: {str(e)}")
                                    continue
                        
                        if not revenue_metric_found:
                            st.warning("Revenue metric not found in forecast data")
                        
                        # 2. Total Adherence Adjusted Volume Comparison
                        volume_metric_found = False
                        for possible_name in ['Total adherence adjusted Volume', 'Total Adherence Adjusted Volume', 'Total Volume']:
                            if possible_name in df_brand_original.index and len(year_cols) > 0:
                                try:
                                    volume_original = df_brand_original.loc[possible_name, year_cols].astype(float)
                                    volume_events = df_brand_with_events.loc[possible_name, year_cols].astype(float)
                                    volume_metric_found = True
                                    
                                    fig_volume = go.Figure()
                                    fig_volume.add_trace(go.Scatter(
                                        x=year_cols,
                                        y=volume_original,
                                        name='Without Events',
                                        mode='lines+markers',
                                        line=dict(color='#93C5FD', width=3),
                                        marker=dict(size=8),
                                        fill='tozeroy',
                                        fillcolor='rgba(147, 197, 253, 0.2)'
                                    ))
                                    fig_volume.add_trace(go.Scatter(
                                        x=year_cols,
                                        y=volume_events,
                                        name='With Events',
                                        mode='lines+markers',
                                        line=dict(color='#34D399', width=3),
                                        marker=dict(size=8),
                                        fill='tozeroy',
                                        fillcolor='rgba(52, 211, 153, 0.2)'
                                    ))
                                    fig_volume.update_layout(
                                        title='Total Adherence Adjusted Volume: Impact of Events',
                                        xaxis_title='Year',
                                        yaxis_title='Volume (Doses)',
                                        hovermode='x unified',
                                        height=400
                                    )
                                    st.plotly_chart(fig_volume, use_container_width=True)
                                    break
                                except Exception as e:
                                    st.error(f"Error plotting volume for {possible_name}: {str(e)}")
                                    continue
                        
                        if not volume_metric_found:
                            st.warning("Volume metric not found in forecast data")
                        
                        # 3. Market Share Comparison
                        market_share_metric_found = False
                        for possible_name in ['Brand X Patient Share', f'{brand_name} Patient Share', 'Patient Share', 'Market Share']:
                            if possible_name in df_brand_original.index and len(year_cols) > 0:
                                try:
                                    share_original = df_brand_original.loc[possible_name, year_cols].astype(float)
                                    share_events = df_brand_with_events.loc[possible_name, year_cols].astype(float)
                                    market_share_metric_found = True
                                    
                                    fig_share = go.Figure()
                                    fig_share.add_trace(go.Scatter(
                                        x=year_cols,
                                        y=share_original * 100,  # Convert to percentage
                                        name='Without Events',
                                        mode='lines+markers',
                                        line=dict(color='#93C5FD', width=3),
                                        marker=dict(size=8)
                                    ))
                                    fig_share.add_trace(go.Scatter(
                                        x=year_cols,
                                        y=share_events * 100,  # Convert to percentage
                                        name='With Events',
                                        mode='lines+markers',
                                        line=dict(color='#34D399', width=3),
                                        marker=dict(size=8)
                                    ))
                                    fig_share.update_layout(
                                        title=f'{brand_name} Market Share: Impact of Events',
                                        xaxis_title='Year',
                                        yaxis_title='Market Share (%)',
                                        hovermode='x unified',
                                        height=400
                                    )
                                    st.plotly_chart(fig_share, use_container_width=True)
                                    break
                                except Exception as e:
                                    st.error(f"Error plotting market share for {possible_name}: {str(e)}")
                                    continue
                        
                        # 4. Brand Patients Comparison (if available)
                        patients_metric_found = False
                        for possible_name in ['Patients on Brand X', f'{brand_name} Patients', 'Brand Patients', f'Patients on {brand_name}']:
                            if possible_name in df_brand_original.index and len(year_cols) > 0:
                                try:
                                    patients_original = df_brand_original.loc[possible_name, year_cols].astype(float)
                                    patients_events = df_brand_with_events.loc[possible_name, year_cols].astype(float)
                                    patients_metric_found = True
                                    
                                    fig_patients = go.Figure()
                                    fig_patients.add_trace(go.Scatter(
                                        x=year_cols,
                                        y=patients_original,
                                        name='Without Events',
                                        mode='lines+markers',
                                        line=dict(color='#93C5FD', width=3),
                                        marker=dict(size=8)
                                    ))
                                    fig_patients.add_trace(go.Scatter(
                                        x=year_cols,
                                        y=patients_events,
                                        name='With Events',
                                        mode='lines+markers',
                                        line=dict(color='#34D399', width=3),
                                        marker=dict(size=8)
                                    ))
                                    fig_patients.update_layout(
                                        title=f'{brand_name} Patients: Impact of Events',
                                        xaxis_title='Year',
                                        yaxis_title='Patients',
                                        hovermode='x unified',
                                        height=400
                                    )
                                    st.plotly_chart(fig_patients, use_container_width=True)
                                    break
                                except Exception as e:
                                    st.error(f"Error plotting patients for {possible_name}: {str(e)}")
                                    continue
                        
                        st.markdown("---")
                        st.markdown("#### 🔬 Epidemiology with Events")
                        df_epi_events_display = df_epi_with_events.copy()
                        
                        # Format epidemiology data
                        for idx in df_epi_events_display.index:
                            if 'rate' in idx.lower() or '%' in idx:
                                df_epi_events_display.loc[idx] = df_epi_events_display.loc[idx].apply(lambda x: f'{float(x)*100:.2f}%' if pd.notna(x) and x != '' else '')
                            else:
                                df_epi_events_display.loc[idx] = df_epi_events_display.loc[idx].apply(lambda x: f'{float(x):,.0f}' if pd.notna(x) and x != '' else '')
                        
                        st.dataframe(df_epi_events_display, use_container_width=True)
                        
                        st.markdown("---")
                        st.markdown(f"#### 💊 {brand_name} Forecast with Events")
                        df_brand_events_display = df_brand_with_events.copy()
                        
                        # Format brand data with proper volume/dose handling
                        for idx in df_brand_events_display.index:
                            # Check for volume/dose FIRST (before percentage checks)
                            if 'volume' in idx.lower() or ('dose' in idx.lower() and 'price' not in idx.lower()):
                                df_brand_events_display.loc[idx] = df_brand_events_display.loc[idx].apply(lambda x: f'{float(x):,.0f}' if pd.notna(x) and x != '' else '')
                            # Then check for rates/shares (but NOT if it's a volume/dose metric)
                            elif 'share' in idx.lower() or ('rate' in idx.lower() and 'volume' not in idx.lower()):
                                df_brand_events_display.loc[idx] = df_brand_events_display.loc[idx].apply(lambda x: f'{float(x)*100:.2f}%' if pd.notna(x) and x != '' else '')
                            # Adherence as percentage (but already caught volume above)
                            elif 'adherence' in idx.lower() and 'volume' not in idx.lower():
                                df_brand_events_display.loc[idx] = df_brand_events_display.loc[idx].apply(lambda x: f'{float(x)*100:.2f}%' if pd.notna(x) and x != '' else '')
                            # Revenue
                            elif 'revenue' in idx.lower() or 'billions' in idx.lower():
                                df_brand_events_display.loc[idx] = df_brand_events_display.loc[idx].apply(lambda x: f'${float(x):,.1f}B' if pd.notna(x) and x != '' else '')
                            # Price
                            elif 'price' in idx.lower() or '$' in idx:
                                df_brand_events_display.loc[idx] = df_brand_events_display.loc[idx].apply(lambda x: f'${float(x):,.2f}' if pd.notna(x) and x != '' else '')
                            # Default: whole numbers for patient counts and volumes
                            else:
                                df_brand_events_display.loc[idx] = df_brand_events_display.loc[idx].apply(lambda x: f'{float(x):,.0f}' if pd.notna(x) and x != '' else '')
                        
                        st.dataframe(df_brand_events_display, use_container_width=True)
                    
                    # ===== TAB 3: DOWNLOAD =====
                    with compare_tab3:
                        st.markdown("### 📥 Download Forecast with Events")
                        
                        st.info("""
                        **Download your forecast with all events applied!**
                        
                        This Excel file includes:
                        - ✅ Updated Epidemiology Forecast (with market events)
                        - ✅ Updated Brand Forecast (with brand events)
                        - ✅ Event summary sheets
                        - ✅ All formulas and calculations
                        - ✅ Research citations
                        - ✅ Complete instructions
                        """)
                        
                        # Show event summary before download
                        if has_market_events:
                            st.markdown("#### 🌍 Market Events Summary")
                            market_event_data = []
                            for event in st.session_state['market_events']:
                                impact_display = f"{event['impact_value']*100:+.1f}%" if event['impact_type'] == 'percentage_change' else f"{event['impact_value']:+,.0f}"
                                market_event_data.append({
                                    'Year': event['event_year'],
                                    'Type': event['event_type'],
                                    'Target': event['target_metric'],
                                    'Impact': impact_display
                                })
                            st.dataframe(pd.DataFrame(market_event_data), use_container_width=True, hide_index=True)
                        
                        if has_brand_events:
                            st.markdown("#### 💊 Brand Events Summary")
                            brand_event_data = []
                            for event in st.session_state['brand_events']:
                                brand_event_data.append({
                                    'Year': event['event_year'],
                                    'Type': event.get('event_type', 'Event'),
                                    'New Peak Share': f"{event['post_event_peak_share']*100:.1f}%",
                                    'Curve': event.get('curve_type', 'N/A')
                                })
                            st.dataframe(pd.DataFrame(brand_event_data), use_container_width=True, hide_index=True)
                        
                        st.markdown("---")
                        
                        # Download button using export_complete_model_with_events
                        excel_name_events = f"forecast_with_events_{brand_name.replace(' ', '_')}_{datetime.now().strftime('%Y%m%d_%H%M%S')}.xlsx"
                        
                        try:
                            excel_bytes_events = export_complete_model_with_events(
                                df_epi_with_events,
                                df_brand_with_events,
                                st.session_state['brand_config'],
                                st.session_state.get('ai_assumptions_v2', {}),
                                market_events=st.session_state.get('market_events'),
                                brand_events=st.session_state.get('brand_events'),
                                filename=excel_name_events
                            )
                            
                            st.download_button(
                                '📥 Download Complete Model with Events',
                                data=excel_bytes_events,
                                file_name=excel_name_events,
                                mime='application/vnd.ms-excel',
                                type='primary',
                                use_container_width=True
                            )
                        except Exception as e:
                            st.error(f"Error generating Excel: {str(e)}")
                            import traceback
                            st.code(traceback.format_exc())
        
    
    # ============================================================
    # TAB 6: SENSITIVITY ANALYSIS
    # ============================================================
    with tab_sensitivity:
        st.header("🎯 Sensitivity Analysis Dashboard")
        st.markdown("""
        Analyze how changes in key parameters affect your brand forecast results. 
        Understand which factors have the greatest impact on revenue and patient volumes.
        """)
        
        if 'df_brand_forecast' not in st.session_state or st.session_state['df_brand_forecast'] is None:
            st.info("📊 Please generate a brand forecast in the Brand Analysis tab first.")
        else:
            # Create sub-tabs for different sensitivity analysis types
            sens_tab1, sens_tab2, sens_tab3 = st.tabs([
                "📊 Interactive Analysis",
                "🎯 Impact Ranking",
                "⚠️ Risk Scenarios"
            ])
            
            # Get base forecast data and configuration
            base_df = st.session_state['df_brand_forecast'].copy()
            brand_config = st.session_state.get('brand_config', {})
            
            # Get year columns
            year_cols = [col for col in base_df.columns if str(col).isdigit()]
            
            # === SUB-TAB 1: INTERACTIVE ANALYSIS ===
            with sens_tab1:
                st.subheader("💡 Real-Time Parameter Adjustment")
                st.markdown("*Adjust sliders to see immediate impact on forecast metrics*")
                
                # Get baseline metrics
                try:
                    base_revenue_total = base_df.loc['Revenue in billions', year_cols].astype(float).sum()
                    base_patients_total = base_df.loc['Patients on Brand X', year_cols].astype(float).sum()
                    base_peak_share = base_df.loc['Brand X Patient Share', year_cols].astype(float).max()
                except:
                    st.error("Could not extract baseline metrics from forecast. Please regenerate the brand forecast.")
                    st.stop()
                
                st.markdown("---")
                col1, col2 = st.columns(2)
                
                with col1:
                    st.markdown("**📈 Market Parameters**")
                    peak_share_adj = st.slider(
                        "Peak Market Share Adjustment (%)",
                        min_value=-50.0,
                        max_value=50.0,
                        value=0.0,
                        step=5.0,
                        help="Adjust the peak market share by percentage points (e.g., +10% means peak share increases from 20% to 30%)"
                    )
                    
                    adherence_adj = st.slider(
                        "Adherence Rate Adjustment (%)",
                        min_value=-30.0,
                        max_value=30.0,
                        value=0.0,
                        step=5.0,
                        help="Adjust patient adherence by percentage points"
                    )
                
                with col2:
                    st.markdown("**💰 Financial Parameters**")
                    price_adj = st.slider(
                        "Price Adjustment (%)",
                        min_value=-50.0,
                        max_value=50.0,
                        value=0.0,
                        step=5.0,
                        help="Adjust price per dose as a percentage change"
                    )
                    
                    dose_adj = st.slider(
                        "Annual Dose Adjustment (%)",
                        min_value=-30.0,
                        max_value=30.0,
                        value=0.0,
                        step=5.0,
                        help="Adjust annual doses per patient"
                    )
                
                # Calculate adjusted metrics
                # Market share impact
                adjusted_peak_share = base_peak_share * (1 + peak_share_adj/100)
                share_multiplier = (1 + peak_share_adj/100)
                
                # Adherence impact
                adherence_multiplier = (1 + adherence_adj/100)
                
                # Price and dose impact
                price_multiplier = (1 + price_adj/100)
                dose_multiplier = (1 + dose_adj/100)
                
                # Calculate new totals
                adjusted_patients_total = base_patients_total * share_multiplier * adherence_multiplier
                adjusted_revenue_total = base_revenue_total * share_multiplier * adherence_multiplier * price_multiplier * dose_multiplier
                
                # Display impact summary
                st.markdown("---")
                st.subheader("📊 Impact Summary")
                
                metric_cols = st.columns(4)
                
                with metric_cols[0]:
                    revenue_delta = adjusted_revenue_total - base_revenue_total
                    revenue_pct = (revenue_delta / base_revenue_total * 100) if base_revenue_total > 0 else 0
                    st.metric(
                        "Total Revenue",
                        f"${adjusted_revenue_total:.1f}B",
                        f"{revenue_delta:+.1f}B ({revenue_pct:+.1f}%)"
                    )
                
                with metric_cols[1]:
                    patients_delta = adjusted_patients_total - base_patients_total
                    patients_pct = (patients_delta / base_patients_total * 100) if base_patients_total > 0 else 0
                    st.metric(
                        "Total Patients",
                        f"{adjusted_patients_total:,.0f}",
                        f"{patients_delta:+,.0f} ({patients_pct:+.1f}%)"
                    )
                
                with metric_cols[2]:
                    peak_share_delta = adjusted_peak_share - base_peak_share
                    st.metric(
                        "Peak Market Share",
                        f"{adjusted_peak_share*100:.1f}%",
                        f"{peak_share_delta*100:+.1f}pp"
                    )
                
                with metric_cols[3]:
                    # Calculate revenue per patient
                    base_rev_per_patient = (base_revenue_total * 1e9) / base_patients_total if base_patients_total > 0 else 0
                    adj_rev_per_patient = (adjusted_revenue_total * 1e9) / adjusted_patients_total if adjusted_patients_total > 0 else 0
                    rev_per_patient_delta = adj_rev_per_patient - base_rev_per_patient
                    st.metric(
                        "Revenue per Patient",
                        f"${adj_rev_per_patient:,.0f}",
                        f"${rev_per_patient_delta:+,.0f}"
                    )
                
                # Display adjusted forecast chart
                st.markdown("---")
                st.subheader("📈 Adjusted Revenue Forecast")
                
                # Create adjusted revenue series
                base_revenue_series = base_df.loc['Revenue in billions', year_cols].astype(float)
                adjusted_revenue_series = base_revenue_series * share_multiplier * adherence_multiplier * price_multiplier * dose_multiplier
                
                fig = go.Figure()
                fig.add_trace(go.Bar(
                    x=year_cols,
                    y=base_revenue_series,
                    name='Base Case',
                    marker_color='#93C5FD',
                    opacity=0.7
                ))
                fig.add_trace(go.Bar(
                    x=year_cols,
                    y=adjusted_revenue_series,
                    name='Adjusted',
                    marker_color='#34D399',
                    opacity=0.7
                ))
                fig.update_layout(
                    title='Revenue Comparison: Base vs Adjusted',
                    xaxis_title='Year',
                    yaxis_title='Revenue (Billions $)',
                    barmode='group',
                    height=400,
                    hovermode='x unified'
                )
                st.plotly_chart(fig, use_container_width=True)
                
                # Show assumptions
                with st.expander("📋 View Adjustment Details"):
                    st.markdown("**Applied Adjustments:**")
                    adjustments_data = {
                        'Parameter': ['Peak Market Share', 'Adherence Rate', 'Price per Dose', 'Annual Doses'],
                        'Adjustment': [f"{peak_share_adj:+.1f}%", f"{adherence_adj:+.1f}%", f"{price_adj:+.1f}%", f"{dose_adj:+.1f}%"],
                        'Multiplier': [f"{share_multiplier:.3f}x", f"{adherence_multiplier:.3f}x", f"{price_multiplier:.3f}x", f"{dose_multiplier:.3f}x"]
                    }
                    st.table(pd.DataFrame(adjustments_data))
            
            # === SUB-TAB 2: IMPACT RANKING ===
            with sens_tab2:
                st.subheader("🎯 Parameter Impact Ranking")
                st.markdown("*Shows which parameters have the greatest impact on total revenue*")
                
                # Define parameters to test with realistic ranges
                parameters = [
                    {'name': 'Peak Market Share', 'low': -20, 'high': 20, 'unit': '%'},
                    {'name': 'Price per Dose', 'low': -20, 'high': 20, 'unit': '%'},
                    {'name': 'Adherence Rate', 'low': -15, 'high': 15, 'unit': '%'},
                    {'name': 'Annual Doses', 'low': -15, 'high': 15, 'unit': '%'},
                ]
                
                # Calculate base revenue
                base_revenue_total = base_df.loc['Revenue in billions', year_cols].astype(float).sum()
                
                # Calculate impact for each parameter
                impacts = []
                
                for param in parameters:
                    low_multiplier = 1 + (param['low'] / 100)
                    high_multiplier = 1 + (param['high'] / 100)
                    
                    low_revenue = base_revenue_total * low_multiplier
                    high_revenue = base_revenue_total * high_multiplier
                    
                    low_impact = low_revenue - base_revenue_total
                    high_impact = high_revenue - base_revenue_total
                    
                    impacts.append({
                        'parameter': param['name'],
                        'low_impact': low_impact,
                        'high_impact': high_impact,
                        'range': abs(high_impact - low_impact),
                        'low_pct': param['low'],
                        'high_pct': param['high']
                    })
                
                # Sort by range (total impact)
                impacts_df = pd.DataFrame(impacts).sort_values('range', ascending=True)
                
                # Create tornado chart
                fig = go.Figure()
                
                fig.add_trace(go.Bar(
                    name=f'Downside',
                    y=impacts_df['parameter'],
                    x=impacts_df['low_impact'],
                    orientation='h',
                    marker=dict(color='#EF4444'),
                    text=[f"${x:.1f}B ({p:+.0f}%)" for x, p in zip(impacts_df['low_impact'], impacts_df['low_pct'])],
                    textposition='auto',
                    hovertemplate='<b>%{y}</b><br>Impact: $%{x:.1f}B<extra></extra>'
                ))
                
                fig.add_trace(go.Bar(
                    name=f'Upside',
                    y=impacts_df['parameter'],
                    x=impacts_df['high_impact'],
                    orientation='h',
                    marker=dict(color='#10B981'),
                    text=[f"${x:.1f}B ({p:+.0f}%)" for x, p in zip(impacts_df['high_impact'], impacts_df['high_pct'])],
                    textposition='auto',
                    hovertemplate='<b>%{y}</b><br>Impact: $%{x:.1f}B<extra></extra>'
                ))
                
                fig.update_layout(
                    title='Revenue Sensitivity - Tornado Chart',
                    xaxis_title='Revenue Change from Base Case (Billions $)',
                    yaxis_title='',
                    barmode='overlay',
                    height=400,
                    showlegend=True,
                    legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1)
                )
                
                st.plotly_chart(fig, use_container_width=True)
                
                # Display impact table
                st.markdown("---")
                st.subheader("📋 Detailed Impact Analysis")
                
                display_df = impacts_df.copy()
                display_df['Downside Impact'] = display_df.apply(lambda row: f"${row['low_impact']:.2f}B ({row['low_pct']:+.0f}%)", axis=1)
                display_df['Upside Impact'] = display_df.apply(lambda row: f"${row['high_impact']:.2f}B ({row['high_pct']:+.0f}%)", axis=1)
                display_df['Total Range'] = display_df['range'].apply(lambda x: f"${x:.2f}B")
                display_df = display_df[['parameter', 'Downside Impact', 'Upside Impact', 'Total Range']]
                display_df.columns = ['Parameter', 'Downside Scenario', 'Upside Scenario', 'Total Impact Range']
                display_df = display_df.sort_values('Total Impact Range', ascending=False)
                
                st.dataframe(display_df, use_container_width=True, hide_index=True)
                
                st.info("💡 **Insight**: Parameters with larger ranges have more influence on revenue. Focus risk mitigation efforts on these key drivers.")
            
            # === SUB-TAB 3: RISK SCENARIOS ===
            with sens_tab3:
                st.subheader("⚠️ Risk Scenarios Analysis")
                st.markdown("*Compare Best Case, Base Case, and Worst Case scenarios side-by-side*")
                
                # Define scenarios with realistic assumptions
                st.markdown("### Scenario Definitions")
                
                scenario_col1, scenario_col2, scenario_col3 = st.columns(3)
                
                with scenario_col1:
                    st.markdown("**🔴 Worst Case**")
                    st.caption("Conservative estimates")
                    worst_share = st.number_input("Market Share Adj (%)", value=-20.0, key='worst_share')
                    worst_price = st.number_input("Price Adj (%)", value=-15.0, key='worst_price')
                    worst_adherence = st.number_input("Adherence Adj (%)", value=-15.0, key='worst_adherence')
                    worst_dose = st.number_input("Dose Adj (%)", value=-10.0, key='worst_dose')
                
                with scenario_col2:
                    st.markdown("**🟡 Base Case**")
                    st.caption("Current forecast")
                    st.info("Market Share: 0%")
                    st.info("Price: 0%")
                    st.info("Adherence: 0%")
                    st.info("Dose: 0%")
                
                with scenario_col3:
                    st.markdown("**🟢 Best Case**")
                    st.caption("Optimistic estimates")
                    best_share = st.number_input("Market Share Adj (%)", value=20.0, key='best_share')
                    best_price = st.number_input("Price Adj (%)", value=10.0, key='best_price')
                    best_adherence = st.number_input("Adherence Adj (%)", value=15.0, key='best_adherence')
                    best_dose = st.number_input("Dose Adj (%)", value=10.0, key='best_dose')
                
                # Calculate scenarios
                base_revenue_series = base_df.loc['Revenue in billions', year_cols].astype(float)
                base_patients_series = base_df.loc['Patients on Brand X', year_cols].astype(float)
                
                scenarios = {
                    'Worst Case': {
                        'share': worst_share,
                        'price': worst_price,
                        'adherence': worst_adherence,
                        'dose': worst_dose,
                        'color': '#EF4444'
                    },
                    'Base Case': {
                        'share': 0,
                        'price': 0,
                        'adherence': 0,
                        'dose': 0,
                        'color': '#F59E0B'
                    },
                    'Best Case': {
                        'share': best_share,
                        'price': best_price,
                        'adherence': best_adherence,
                        'dose': best_dose,
                        'color': '#10B981'
                    }
                }
                
                scenario_results = {}
                
                for scenario_name, params in scenarios.items():
                    multiplier = (1 + params['share']/100) * (1 + params['price']/100) * (1 + params['adherence']/100) * (1 + params['dose']/100)
                    revenue_series = base_revenue_series * multiplier
                    patients_series = base_patients_series * (1 + params['share']/100) * (1 + params['adherence']/100)
                    
                    scenario_results[scenario_name] = {
                        'revenue': revenue_series,
                        'patients': patients_series,
                        'total_revenue': revenue_series.sum(),
                        'total_patients': patients_series.sum(),
                        'color': params['color']
                    }
                
                # Display summary metrics
                st.markdown("---")
                st.subheader("📊 Scenario Comparison Summary")
                summary_cols = st.columns(3)
                
                for idx, (scenario_name, results) in enumerate(scenario_results.items()):
                    with summary_cols[idx]:
                        st.markdown(f"**{scenario_name}**")
                        
                        # Calculate delta from base
                        if scenario_name != 'Base Case':
                            base_total_rev = scenario_results['Base Case']['total_revenue']
                            delta_rev = results['total_revenue'] - base_total_rev
                            delta_rev_pct = (delta_rev / base_total_rev * 100) if base_total_rev > 0 else 0
                            
                            st.metric(
                                "Total Revenue",
                                f"${results['total_revenue']:.1f}B",
                                f"{delta_rev:+.1f}B ({delta_rev_pct:+.1f}%)"
                            )
                            
                            base_total_pat = scenario_results['Base Case']['total_patients']
                            delta_pat = results['total_patients'] - base_total_pat
                            delta_pat_pct = (delta_pat / base_total_pat * 100) if base_total_pat > 0 else 0
                            
                            st.metric(
                                "Total Patients",
                                f"{results['total_patients']:,.0f}",
                                f"{delta_pat:+,.0f} ({delta_pat_pct:+.1f}%)"
                            )
                        else:
                            st.metric("Total Revenue", f"${results['total_revenue']:.1f}B")
                            st.metric("Total Patients", f"{results['total_patients']:,.0f}")
                
                # Comparison charts
                st.markdown("---")
                st.subheader("📈 Revenue Comparison Over Time")
                
                fig_revenue = go.Figure()
                
                for scenario_name, results in scenario_results.items():
                    fig_revenue.add_trace(go.Scatter(
                        x=year_cols,
                        y=results['revenue'],
                        mode='lines+markers',
                        name=scenario_name,
                        line=dict(color=results['color'], width=3),
                        marker=dict(size=8)
                    ))
                
                fig_revenue.update_layout(
                    xaxis_title='Year',
                    yaxis_title='Revenue (Billions $)',
                    height=450,
                    hovermode='x unified',
                    legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1)
                )
                
                st.plotly_chart(fig_revenue, use_container_width=True)
                
                # Patient volume comparison
                st.subheader("👥 Patient Volume Comparison")
                
                fig_patients = go.Figure()
                
                for scenario_name, results in scenario_results.items():
                    fig_patients.add_trace(go.Scatter(
                        x=year_cols,
                        y=results['patients'],
                        mode='lines+markers',
                        name=scenario_name,
                        line=dict(color=results['color'], width=3),
                        marker=dict(size=8),
                        fill='tonexty' if scenario_name == 'Best Case' else None,
                        fillcolor=f"rgba{tuple(list(int(results['color'][i:i+2], 16) for i in (1, 3, 5)) + [0.1])}"
                    ))
                
                fig_patients.update_layout(
                    xaxis_title='Year',
                    yaxis_title='Patients',
                    height=450,
                    hovermode='x unified',
                    legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1)
                )
                
                st.plotly_chart(fig_patients, use_container_width=True)
                
                # Export scenarios
                st.markdown("---")
                st.subheader("📥 Export Analysis")
                
                # Create export dataframe
                export_data = {'Year': year_cols}
                for scenario_name, results in scenario_results.items():
                    export_data[f'{scenario_name} - Revenue ($B)'] = results['revenue'].values
                    export_data[f'{scenario_name} - Patients'] = results['patients'].values
                
                export_df = pd.DataFrame(export_data)
                
                buffer = BytesIO()
                with pd.ExcelWriter(buffer, engine='xlsxwriter') as writer:
                    export_df.to_excel(writer, sheet_name='Scenario Comparison', index=False)
                    
                    # Add summary sheet
                    summary_data = []
                    for scenario_name, results in scenario_results.items():
                        summary_data.append({
                            'Scenario': scenario_name,
                            'Total Revenue ($B)': f"{results['total_revenue']:.2f}",
                            'Total Patients': f"{results['total_patients']:,.0f}",
                            'Avg Annual Revenue ($B)': f"{results['total_revenue']/len(year_cols):.2f}",
                            'Avg Annual Patients': f"{results['total_patients']/len(year_cols):,.0f}"
                        })
                    pd.DataFrame(summary_data).to_excel(writer, sheet_name='Summary', index=False)
                
                st.download_button(
                    label="📥 Download Risk Scenarios (Excel)",
                    data=buffer.getvalue(),
                    file_name=f"risk_scenarios_analysis_{datetime.now().strftime('%Y%m%d')}.xlsx",
                    mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
                    use_container_width=True
                )
                
                st.success("✅ Sensitivity analysis complete. Use these insights to understand your forecast's key drivers and risks.")
    
    # ============================================================
    # TAB 7: SCENARIO MANAGEMENT
    # ============================================================
    with tab_scenarios:
        st.header('🔄 Scenario Management')
        
        st.markdown("""
        Create and compare multiple forecasting scenarios with different assumptions.
        
        **Workflow:**
        1. Generate a complete forecast (epidemiology + brand)
        2. Save it as a scenario
        3. Modify parameters (e.g., change peak share, price, uptake)
        4. Generate and save as a new scenario
        5. Compare scenarios side-by-side
        
        Save this forecast as a scenario, then modify parameters to create alternative scenarios and compare them side-by-side.
        """)
        
        # Initialize saved scenarios
        if 'saved_scenarios' not in st.session_state:
            st.session_state['saved_scenarios'] = {}
        
        # Create tabs for scenario workflow
        scenario_tab1, scenario_tab2, scenario_tab3 = st.tabs([
            "Save Current Scenario",
            "Manage Saved Scenarios", 
            "Compare Scenarios"
        ])
        
        # ===== TAB: SAVE SCENARIO =====
        with scenario_tab1:
            st.subheader("Save Current Forecast as Scenario")
            
            # Check if there's data to save
            has_epi = 'df_metrics_custom' in st.session_state or 'df_metrics_base' in st.session_state
            has_brand = 'df_brand_forecast' in st.session_state
            
            if not has_epi:
                st.warning("⚠️ No epidemiology forecast found. Please generate a base forecast first.")
            elif not has_brand:
                st.info("ℹ️ You can save just the epidemiology forecast, or generate a brand forecast first for a complete scenario.")
            
            if has_epi:
                col1, col2 = st.columns([3, 1])
                
                with col1:
                    scenario_name = st.text_input(
                        "Scenario Name",
                        value=f"Scenario_{len(st.session_state.get('saved_scenarios', {})) + 1}",
                        key='save_scenario_name',
                        help="Give this forecast a descriptive name (e.g., 'Base Case', 'High Share - Premium Price')"
                    )
                    
                    scenario_description = st.text_area(
                        "Description (optional)",
                        placeholder="E.g., '35% peak share, $16 price point, 7-year medium uptake, 68% adherence'",
                        key='save_scenario_desc',
                        height=80
                    )
                
                with col2:
                    st.markdown("### Current Config")
                    if has_brand:
                        brand_config = st.session_state.get('brand_config', {})
                        st.info(f"""
                        **Brand:** {brand_config.get('brand_name', 'N/A')}  
                        **Peak Share:** {brand_config.get('peak_share', 0)*100:.0f}%  
                        **Price:** ${brand_config.get('price_per_dose', 0):.2f}  
                        **Uptake:** {brand_config.get('uptake_type', 'N/A')}  
                        **Adherence:** {brand_config.get('adherence', 0)*100:.0f}%
                        """)
                    else:
                        ai_assumptions = st.session_state.get('ai_assumptions_v2', {})
                        known_values = ai_assumptions.get('known_values', {})
                        st.info(f"""
                        **Disease:** {ai_assumptions.get('disease_name', 'N/A')}  
                        **Geography:** {ai_assumptions.get('geography', 'N/A')}  
                        **Prevalence:** {known_values.get('prevalence', 0)*100:.1f}%  
                        **Diagnosed:** {known_values.get('diagnosed', 0)*100:.1f}%  
                        **Treated:** {known_values.get('treated', 0)*100:.1f}%
                        """)
                
                if st.button("💾 Save This Scenario", type="primary", use_container_width=True):
                    if scenario_name.strip():
                        if save_scenario(scenario_name, scenario_description):
                            st.success(f"✅ Scenario '{scenario_name}' saved successfully!")
                            st.info("""
                            **Next Steps:**
                            1. Go to 'Brand Analysis' tab
                            2. Modify parameters (e.g., change peak share to 25%)
                            3. Click 'Generate Brand Forecast' again
                            4. Return here and save as a new scenario
                            5. Use 'Compare Scenarios' tab to analyze differences
                            """)
                            st.rerun()
                    else:
                        st.error("Please provide a scenario name")
        
        # ===== TAB: MANAGE SCENARIOS =====
        with scenario_tab2:
            st.subheader("📂 Saved Scenarios")
            
            if not st.session_state.get('saved_scenarios'):
                st.info("""
                No scenarios saved yet.
                
                **To create scenarios:**
                1. Generate a complete forecast (with or without brand)
                2. Go to 'Save Current Scenario' tab
                3. Give it a name and save
                4. Modify parameters and generate a new forecast
                5. Save again with a different name
                """)
            else:
                st.markdown(f"**Total Scenarios:** {len(st.session_state['saved_scenarios'])}")
                
                # Display scenarios in expandable cards
                for scenario_name, scenario_data in st.session_state['saved_scenarios'].items():
                    with st.expander(f"📊 {scenario_name}", expanded=False):
                        col1, col2, col3 = st.columns([2, 2, 1])
                        
                        with col1:
                            st.markdown("**Basic Info**")
                            st.write(f"**Created:** {scenario_data['timestamp']}")
                            if scenario_data.get('description'):
                                st.write(f"**Description:** {scenario_data['description']}")
                        
                        with col2:
                            if scenario_data['brand_config']:
                                st.markdown("**Brand Configuration**")
                                config = scenario_data['brand_config']
                                st.write(f"**Brand:** {config['brand_name']}")
                                st.write(f"**Launch Year:** {config['launch_year']}")
                                st.write(f"**Peak Share:** {config['peak_share']*100:.1f}%")
                                st.write(f"**Price/Dose:** ${config['price_per_dose']:.2f}")
                                st.write(f"**Uptake:** {config['uptake_type']}")
                                st.write(f"**Adherence:** {config['adherence']*100:.1f}%")
                            else:
                                st.markdown("**Epidemiology Only**")
                                if scenario_data.get('ai_assumptions'):
                                    ai = scenario_data['ai_assumptions']
                                    st.write(f"**Disease:** {ai.get('disease_name', 'N/A')}")
                                    st.write(f"**Geography:** {ai.get('geography', 'N/A')}")
                        
                        with col3:
                            st.markdown("**Actions**")
                            
                            if st.button("📂 Load", key=f"load_{scenario_name}", use_container_width=True):
                                if load_scenario(scenario_name):
                                    st.success(f"Loaded '{scenario_name}'")
                                    st.info("Scroll up to view the loaded forecast. You can modify and save as a new scenario.")
                                    st.rerun()
                            
                            if st.button("🗑️ Delete", key=f"delete_{scenario_name}", use_container_width=True):
                                del st.session_state['saved_scenarios'][scenario_name]
                                st.success(f"Deleted '{scenario_name}'")
                                st.rerun()
                
                # Bulk actions
                st.markdown("---")
                col1, col2 = st.columns([1, 3])
                
                with col1:
                    if st.button("🗑️ Clear All Scenarios", type="secondary"):
                        st.session_state['saved_scenarios'] = {}
                        st.success("All scenarios cleared")
                        st.rerun()
        
        # ===== TAB: COMPARE SCENARIOS =====
        with scenario_tab3:
            st.subheader("📊 Compare Scenarios Side-by-Side")
            
            if not st.session_state.get('saved_scenarios'):
                st.info("""
                No scenarios available for comparison.
                
                Save at least 2 scenarios to use this feature.
                """)
            elif len(st.session_state['saved_scenarios']) < 2:
                st.warning("You need at least 2 scenarios to compare. Create more scenarios in the 'Save Current Scenario' tab.")
            else:
                st.markdown("""
                Select multiple scenarios to compare revenue, patient volumes, and market share across different assumptions.
                """)
                
                # Scenario selection
                scenario_names = list(st.session_state['saved_scenarios'].keys())
                selected_scenarios = st.multiselect(
                    "Select Scenarios to Compare",
                    options=scenario_names,
                    default=scenario_names[:min(3, len(scenario_names))],
                    help="Select 2 or more scenarios"
                )
                
                if len(selected_scenarios) < 2:
                    st.info("Please select at least 2 scenarios to compare")
                else:
                    # Generate comparison
                    comparison = compare_scenarios(selected_scenarios)
                    
                    if comparison and comparison['scenarios']:
                        # Configuration comparison table
                        st.markdown("### ⚙️ Configuration Comparison")
                        
                        config_comparison = []
                        for i, name in enumerate(comparison['scenarios']):
                            config = comparison['configs'][i]
                            if config:  # Has brand config
                                config_comparison.append({
                                    'Scenario': name,
                                    'Brand': config['brand_name'],
                                    'Launch Year': config['launch_year'],
                                    'Peak Share': f"{config['peak_share']*100:.1f}%",
                                    'Uptake Curve': config['uptake_type'],
                                    'Adherence': f"{config['adherence']*100:.1f}%",
                                    'Annual Dose': config['annual_dose'],
                                    'Price/Dose': f"${config['price_per_dose']:.2f}"
                                })
                        
                        if config_comparison:
                            df_config = pd.DataFrame(config_comparison)
                            st.dataframe(df_config, use_container_width=True, hide_index=True)
                            
                            st.markdown("---")
                            
                            # Visual comparisons
                            st.markdown("### 📈 Visual Comparison")
                            
                            years = comparison.get('years', [])
                            
                            if years and comparison.get('revenue_data'):
                                # Revenue comparison - Grouped Bar Chart
                                st.markdown("#### 💰 Revenue Comparison")
                                
                                import plotly.graph_objects as go
                                fig_revenue = go.Figure()
                                
                                for scenario_name in comparison['scenarios']:
                                    if scenario_name in comparison['revenue_data']:
                                        fig_revenue.add_trace(go.Bar(
                                            name=scenario_name,
                                            x=years,
                                            y=comparison['revenue_data'][scenario_name],
                                            text=[f"${v:.1f}B" for v in comparison['revenue_data'][scenario_name]],
                                            textposition='outside'
                                        ))
                                
                                fig_revenue.update_layout(
                                    barmode='group',
                                    title='Revenue Comparison Across Scenarios',
                                    xaxis_title='Year',
                                    yaxis_title='Revenue ($ Billions)',
                                    height=500,
                                    legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1)
                                )
                                st.plotly_chart(fig_revenue, use_container_width=True)
                                
                                # Patient volume and share comparisons
                                col1, col2 = st.columns(2)
                                
                                with col1:
                                    if comparison.get('patient_data'):
                                        st.markdown("#### 👥 Patient Volume Comparison")
                                        fig_patients = go.Figure()
                                        
                                        for scenario_name in comparison['scenarios']:
                                            if scenario_name in comparison['patient_data']:
                                                fig_patients.add_trace(go.Scatter(
                                                    name=scenario_name,
                                                    x=years,
                                                    y=comparison['patient_data'][scenario_name],
                                                    mode='lines+markers',
                                                    line=dict(width=3),
                                                    marker=dict(size=8)
                                                ))
                                        
                                        fig_patients.update_layout(
                                            title='Patients on Brand Comparison',
                                            xaxis_title='Year',
                                            yaxis_title='Patients',
                                            height=400,
                                            legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1)
                                        )
                                        st.plotly_chart(fig_patients, use_container_width=True)
                                
                                with col2:
                                    if comparison.get('share_data'):
                                        st.markdown("#### 📊 Market Share Comparison")
                                        fig_share = go.Figure()
                                        
                                        for scenario_name in comparison['scenarios']:
                                            if scenario_name in comparison['share_data']:
                                                share_pct = [v * 100 for v in comparison['share_data'][scenario_name]]
                                                fig_share.add_trace(go.Scatter(
                                                    name=scenario_name,
                                                    x=years,
                                                    y=share_pct,
                                                    mode='lines+markers',
                                                    line=dict(width=3),
                                                    marker=dict(size=8)
                                                ))
                                        
                                        fig_share.update_layout(
                                            title='Market Share Evolution',
                                            xaxis_title='Year',
                                            yaxis_title='Market Share (%)',
                                            height=400,
                                            legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1)
                                        )
                                        st.plotly_chart(fig_share, use_container_width=True)
                                
                                # Summary metrics
                                st.markdown("---")
                                st.markdown("### 📊 Summary Metrics")
                                
                                summary_data = []
                                for scenario_name in comparison['scenarios']:
                                    if scenario_name in comparison['revenue_data']:
                                        total_revenue = sum(comparison['revenue_data'][scenario_name])
                                        peak_patients = max(comparison['patient_data'].get(scenario_name, [0]))
                                        peak_share = max(comparison['share_data'].get(scenario_name, [0])) * 100
                                        
                                        summary_data.append({
                                            'Scenario': scenario_name,
                                            'Total Revenue': f"${total_revenue:.1f}B",
                                            'Peak Patients': f"{peak_patients:,.0f}",
                                            'Peak Market Share': f"{peak_share:.1f}%"
                                        })
                                
                                if summary_data:
                                    df_summary = pd.DataFrame(summary_data)
                                    st.dataframe(df_summary, use_container_width=True, hide_index=True)
                        else:
                            st.warning("Selected scenarios don't have brand forecasts to compare. Please select scenarios with brand data.")

# Entry point
if __name__ == '__main__':
    main()
