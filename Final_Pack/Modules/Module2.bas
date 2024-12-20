Attribute VB_Name = "Module2"
Sub InsertIcon()
    Dim ws As Worksheet
    Dim iconPath As String
    Dim iconLeft As Double, iconTop As Double

     Set the path to your icon file
    iconPath = "D:\Axtria\credit.png"

     Set the worksheet where you want to insert the icon
    Set ws = ThisWorkbook.Sheets("Sheet1")

     Set the position for the icon (left and top coordinates)
    iconLeft = 100
    iconTop = 100

     Insert the icon
    With ws.Pictures.Insert(iconPath)
        .Left = iconLeft
        .Top = iconTop
        .Placement = 1  Move and size with cells
        .PrintObject = True
    End With
End Sub
