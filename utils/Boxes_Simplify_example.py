import json
import os
from dataclasses import dataclass
from itertools import combinations
from typing import List
import matplotlib.pyplot as plt
import numpy as np
from PIL import Image, ImageFont, ImageDraw
from matplotlib.backend_bases import MouseButton
from matplotlib.patches import Rectangle
from sklearn.cluster import AgglomerativeClustering


class Node:
    PreviousLeftNode: 'Node'
    PreviousRightNode: 'Node'
    Combined: True
    Y2: int
    Y1: int
    X2: int
    X1: int
    SubNodes: []
    CombinedType: -1

    def __init__(self, X1: int, X2: int, Y1: int, Y2: int):
        self.Y2 = Y2
        self.Y1 = Y1
        self.X2 = X2
        self.X1 = X1
        self.Combined = False
        self.SubNodes = []
        self.CombinedType = -1
        self.PreviousLeftNode = None
        self.PreviousRightNode = None

    def AddSubNode(self, X1: int, X2: int, Y1: int, Y2: int):
        self.SubNodes.append(Node(X1, X2, Y1, Y2))

    def Copy(self):
        ReturnNode = Node(self.X1, self.X2, self.Y1, self.Y2)
        ReturnNode.Combined = self.Combined
        ReturnNode.PreviousLeftNode = self.PreviousLeftNode
        ReturnNode.PreviousRightNode = self.PreviousRightNode
        ReturnNode.SubNodes = self.SubNodes
        ReturnNode.CombinedType = self.CombinedType
        return ReturnNode


def getOrigin(Node):
    if Node is None:
        return []
    else:
        if (Node.PreviousLeftNode == None and Node.PreviousRightNode == None):
            return [Node]
        else:
            return [] + getOrigin(Node.PreviousLeftNode) + getOrigin(Node.PreviousRightNode)


def PointsToNodesConversion(Points, Indices):
    ListOfNodes = []
    I = 0
    for Indice in Indices:
        if Points[Indice]['Y2'] > Points[Indice]['Y1']:
            Y2 = Points[Indice]['Y2']
            Y1 = Points[Indice]['Y1']
        else:
            Y2 = Points[Indice]['Y1']
            Y1 = Points[Indice]['Y2']
        if Points[Indice]['X2'] > Points[Indice]['X1']:
            X2 = Points[Indice]['X2']
            X1 = Points[Indice]['X1']
        else:
            X2 = Points[Indice]['X1']
            X1 = Points[Indice]['X2']
        ListOfNodes.append(Node(X1, X2, Y1, Y2))
        I += 1
    return ListOfNodes


def ConvertSubPointsToPoints(ComplexNode, Starting=0):
    NodeList = getOrigin(ComplexNode)
    IndexList = []
    for i in range(len(NodeList)):
        IndexList.append(i + Starting)
    return NodeList, IndexList


def SubPointsToSubPoints(NewTextBoxesA, NewTextBoxesB, XDeviation, YDeviation):
    Points = []
    for NewtextBox in NewTextBoxesA:
        Points.append([NewtextBox.X1, NewtextBox.Y1])
        Points.append([NewtextBox.X1, NewtextBox.Y2])
        Points.append([NewtextBox.X2, NewtextBox.Y1])
        Points.append([NewtextBox.X2, NewtextBox.Y2])

    for NewtextBox in NewTextBoxesB:
        Points.append([NewtextBox.X1, NewtextBox.Y1])
        Points.append([NewtextBox.X1, NewtextBox.Y2])
        Points.append([NewtextBox.X2, NewtextBox.Y1])
        Points.append([NewtextBox.X2, NewtextBox.Y2])

    AuxPoints = Points.copy()
    XValues = list(set([elem[0] for elem in AuxPoints]))
    XValues.sort()
    Points = []
    for XIndex in range(len(XValues)):
        PointValue = []
        for Point in AuxPoints:
            if Point[0] == XValues[XIndex]:
                PointValue.append(Point[1])
        PointValue.sort()
        Point = {"XValue": int, "AvailableY": [[]]}
        Point["XValue"] = XValues[XIndex]
        CombList = list(combinations(list(range(len(PointValue))), 2))
        UsableYSpaced = []
        used_indexes = []
        for Comb in CombList:
            if not (Comb[0] in used_indexes or Comb[1] in used_indexes):
                UsableYSpaced.append([PointValue[Comb[0]], PointValue[Comb[1]]])
                used_indexes.append(Comb[0])
                used_indexes.append(Comb[1])

        def SimplifyPoints(UsableYSpaced):
            CombList = list(combinations(list(range(len(UsableYSpaced))), 2))
            for Comb in CombList:
                if UsableYSpaced[Comb[0]][0] <= UsableYSpaced[Comb[1]][0] <= UsableYSpaced[Comb[0]][1] and \
                        UsableYSpaced[Comb[1]][0] <= UsableYSpaced[Comb[0]][1] <= UsableYSpaced[Comb[1]][1]:
                    aux = [UsableYSpaced[Comb[0]][0], UsableYSpaced[Comb[0]][1], UsableYSpaced[Comb[1]][0],
                           UsableYSpaced[Comb[1]][1]]
                    UsableYSpaced[Comb[0]] = [min(aux), max(aux)]
                    UsableYSpaced.pop(Comb[1])
                    return SimplifyPoints(UsableYSpaced)
            return UsableYSpaced

        Point["AvailableY"] = SimplifyPoints(UsableYSpaced)
        ToRemoveList = []
        for aux in Point["AvailableY"]:
            if aux[0] == aux[1]:
                ToRemoveList.append(aux)
        for aux in ToRemoveList:
            Point["AvailableY"].pop(Point["AvailableY"].index(aux))
        Points.append(Point)
    Points.sort(key=lambda x: x["XValue"], reverse=False)
    YValues = list(set([elem[1] for elem in AuxPoints]))
    YValues.sort()

    TestLines = []
    for i in range(1, len(YValues)):
        TestLines.append([YValues[i - 1], YValues[i]])
    Avgheight = [elem[1]-elem[0] for elem in TestLines]
    Testvals = [sum(elem) / len(elem) for elem in TestLines]

    for i_elem in range(len(TestLines)):
        if Avgheight[i_elem] <= 3 * YDeviation:
            TestLines.pop(TestLines.index(TestLines[i_elem]))

    TextBox = Node(min(XValues), max(XValues), min(YValues), max(YValues))
    for TestIndex in range(len(TestLines)):
        XPoints = {"YIntreval": TestLines[TestIndex], "XIntreval": []}
        Previous_Entry_Geometry = False
        Entry_Geometry = False
        for Point in Points:
            for Interval in Point["AvailableY"]:
                if Interval[0] <= Testvals[TestIndex] <= Interval[1]:
                    Entry_Geometry = not (Entry_Geometry)
                    break
            if Entry_Geometry and not Previous_Entry_Geometry:
                AuxVal = Point["XValue"]
                Previous_Entry_Geometry = Entry_Geometry
            if Previous_Entry_Geometry and not Entry_Geometry:
                XPoints["XIntreval"].append([AuxVal, Point["XValue"]])
                Previous_Entry_Geometry = Entry_Geometry

        def SimplifyPoints(UsableYSpaced):
            CombList = list(combinations(list(range(len(UsableYSpaced))), 2))
            for Comb in CombList:
                if UsableYSpaced[Comb[0]][0] <= UsableYSpaced[Comb[1]][0] <= UsableYSpaced[Comb[0]][1] and \
                        UsableYSpaced[Comb[1]][0] <= UsableYSpaced[Comb[0]][1] <= UsableYSpaced[Comb[1]][1]:
                    aux = [UsableYSpaced[Comb[0]][0], UsableYSpaced[Comb[0]][1], UsableYSpaced[Comb[1]][0],
                           UsableYSpaced[Comb[1]][1]]
                    UsableYSpaced[Comb[0]] = [min(aux), max(aux)]
                    UsableYSpaced.pop(Comb[1])
                    return SimplifyPoints(UsableYSpaced)
            return UsableYSpaced

        XPoints["XIntreval"] = SimplifyPoints(XPoints["XIntreval"])
        for Intreval in XPoints["XIntreval"]:
            TextBox.SubNodes.append(
                Node(Intreval[0], Intreval[1], XPoints["YIntreval"][0], XPoints["YIntreval"][1]))
        TextBox.Combined = True
    return TextBox


def PointComparison(TextBoxes, XMD, YMD, IndexListIn=None, CombListIn=None):
    if len(TextBoxes) <= 1:
        return TextBoxes
    YDeviation = YMD
    XDeviation = XMD
    if not IndexListIn:
        IndexListIn = list(range(len(TextBoxes)))
    if CombListIn:
        CombListAux = CombListIn.copy()
    else:
        CombListIn = list(combinations(IndexListIn, 2))
        CombListAux = CombListIn.copy()
    for Comb in CombListIn:
        TextBoxA = TextBoxes[Comb[0]]
        TextBoxB = TextBoxes[Comb[1]]
        VerticalInside = False
        HorizontalInside = False
        VerticalIntersect = False
        HorizontalIntersect = False
        SequenceBound = False
        if (
                TextBoxA.Y1 - YDeviation <= TextBoxB.Y1 <= TextBoxA.Y2 + YDeviation and TextBoxB.Y1 - YDeviation <= TextBoxA.Y2 <= TextBoxB.Y2 + YDeviation) or (
                TextBoxB.Y1 - YDeviation <= TextBoxA.Y1 <= TextBoxB.Y2 + YDeviation and TextBoxA.Y1 - YDeviation <= TextBoxB.Y2 <= TextBoxA.Y2 + YDeviation):
            VerticalIntersect = True
        if (
                TextBoxA.X1 - XDeviation <= TextBoxB.X1 <= TextBoxA.X2 + XDeviation and TextBoxB.X1 - XDeviation <= TextBoxA.X2 <= TextBoxB.X2 + XDeviation) or (
                TextBoxB.X1 - XDeviation <= TextBoxA.X1 <= TextBoxB.X2 + XDeviation and TextBoxA.X1 - XDeviation <= TextBoxB.X2 <= TextBoxA.X2 + XDeviation):
            HorizontalIntersect = True
        if (
                TextBoxA.Y1 - YDeviation <= TextBoxB.Y1 <= TextBoxA.Y2 + YDeviation and TextBoxA.Y1 - YDeviation <= TextBoxB.Y2 <= TextBoxA.Y2 + YDeviation) or (
                TextBoxB.Y1 - YDeviation <= TextBoxA.Y1 <= TextBoxB.Y2 + YDeviation and TextBoxB.Y1 - YDeviation <= TextBoxA.Y2 <= TextBoxB.Y2 + YDeviation):
            VerticalInside = True
        if (
                TextBoxA.X1 - XDeviation <= TextBoxB.X1 <= TextBoxA.X2 + XDeviation and TextBoxA.X1 - XDeviation <= TextBoxB.X2 <= TextBoxA.X2 + XDeviation) or (
                TextBoxB.X1 - XDeviation <= TextBoxA.X1 <= TextBoxB.X2 + XDeviation and TextBoxB.X1 - XDeviation <= TextBoxA.X2 <= TextBoxB.X2 + XDeviation):
            HorizontalInside = True
        if VerticalInside and HorizontalInside:
            if not TextBoxA.Combined and not TextBoxB.Combined:
                Y1List = list((TextBoxA.Y1, TextBoxB.Y1))
                Y2List = list((TextBoxA.Y2, TextBoxB.Y2))
                X1List = list((TextBoxA.X1, TextBoxB.X1))
                X2List = list((TextBoxA.X2, TextBoxB.X2))
                TextBoxes[Comb[0]].Y2 = max(Y2List)
                TextBoxes[Comb[0]].Y1 = min(Y1List)
                TextBoxes[Comb[0]].X2 = max(X2List)
                TextBoxes[Comb[0]].X1 = min(X1List)
                TextBoxes[Comb[0]].Combined = False
                TextBoxes[Comb[0]].CombinedType = 1
                TextBoxes[Comb[0]].PreviousLeftNode = None
                TextBoxes[Comb[0]].PreviousRightNode = None
                TextBoxes.pop(Comb[1])
                return PointComparison(TextBoxes, XMD, YMD)

        elif HorizontalInside and (VerticalIntersect or VerticalInside):
            if not TextBoxA.Combined and not TextBoxB.Combined:
                Y1List = list((TextBoxA.Y1, TextBoxB.Y1))
                Y2List = list((TextBoxA.Y2, TextBoxB.Y2))
                X1List = list((TextBoxA.X1, TextBoxB.X1))
                X2List = list((TextBoxA.X2, TextBoxB.X2))
                TextBoxes[Comb[0]].Y2 = max(Y2List)
                TextBoxes[Comb[0]].Y1 = min(Y1List)
                TextBoxes[Comb[0]].X2 = sum(X2List)/len(X2List)
                TextBoxes[Comb[0]].X1 = sum(X1List)/len(X1List)
                TextBoxes[Comb[0]].Combined = False
                TextBoxes[Comb[0]].CombinedType = 1
                TextBoxes[Comb[0]].PreviousLeftNode = None
                TextBoxes[Comb[0]].PreviousRightNode = None
                TextBoxes.pop(Comb[1])
                return PointComparison(TextBoxes, XMD, YMD)

        elif VerticalIntersect or HorizontalIntersect:
            if ((TextBoxA.X2 - XDeviation) <= TextBoxB.X1 <= (TextBoxA.X2 + XDeviation) or (TextBoxA.X1 - XDeviation) <= TextBoxB.X2 <= (TextBoxA.X1 + XDeviation)) and ((( TextBoxA.Y1 - YDeviation <= TextBoxB.Y1 <= TextBoxA.Y1 + YDeviation) and ( TextBoxA.Y2 - YDeviation <= TextBoxB.Y2 <= TextBoxA.Y2 + YDeviation)) or (
                                                                                                         (
                                                                                                                 TextBoxB.Y1 - YDeviation <= TextBoxA.Y1 <= TextBoxB.Y1 + YDeviation) and (
                                                                                                                 TextBoxB.Y2 - YDeviation <= TextBoxA.Y2 <= TextBoxB.Y2 + YDeviation))):
                SequenceBound = True
            if SequenceBound:
                if not TextBoxA.Combined and not TextBoxB.Combined:
                    Y1List = list((TextBoxA.Y1, TextBoxB.Y1))
                    Y2List = list((TextBoxA.Y2, TextBoxB.Y2))
                    X1List = list((TextBoxA.X1, TextBoxB.X1))
                    X2List = list((TextBoxA.X2, TextBoxB.X2))
                    TextBoxes[Comb[0]].Y2 = sum(Y2List) / len(Y2List)
                    TextBoxes[Comb[0]].Y1 = sum(Y1List) / len(Y1List)
                    TextBoxes[Comb[0]].X2 = max(X2List)
                    TextBoxes[Comb[0]].X1 = min(X1List)
                    TextBoxes[Comb[0]].Combined = False
                    TextBoxes[Comb[0]].CombinedType = 3
                    TextBoxes[Comb[0]].PreviousLeftNode = None
                    TextBoxes[Comb[0]].PreviousRightNode = None
                    TextBoxes.pop(Comb[1])
                    return PointComparison(TextBoxes, XMD, YMD)
    CombListAux = CombListIn.copy()
    for Comb in CombListIn:
        TextBoxA = TextBoxes[Comb[0]]
        TextBoxB = TextBoxes[Comb[1]]
        VerticalInside = False
        VerticalIntersect = False
        HorizontalIntersect = False
        HorizontalInside = False
        Rightbound = False
        LeftBound = False
        if (
                TextBoxA.Y1 - YDeviation <= TextBoxB.Y1 <= TextBoxA.Y2 + YDeviation and TextBoxB.Y1 - YDeviation <= TextBoxA.Y2 <= TextBoxB.Y2 + YDeviation) or (
                TextBoxB.Y1 - YDeviation <= TextBoxA.Y1 <= TextBoxB.Y2 + YDeviation and TextBoxA.Y1 - YDeviation <= TextBoxB.Y2 <= TextBoxA.Y2 + YDeviation):
            VerticalIntersect = True
        if (
                TextBoxA.X1 - XDeviation <= TextBoxB.X1 <= TextBoxA.X2 + XDeviation and TextBoxB.X1 - XDeviation <= TextBoxA.X2 <= TextBoxB.X2 + XDeviation) or (
                TextBoxB.X1 - XDeviation <= TextBoxA.X1 <= TextBoxB.X2 + XDeviation and TextBoxA.X1 - XDeviation <= TextBoxB.X2 <= TextBoxA.X2 + XDeviation):
            HorizontalIntersect = True
        if (
                TextBoxA.Y1 - YDeviation <= TextBoxB.Y1 <= TextBoxA.Y2 + YDeviation and TextBoxA.Y1 - YDeviation <= TextBoxB.Y2 <= TextBoxA.Y2 + YDeviation) or (
                TextBoxB.Y1 - YDeviation <= TextBoxA.Y1 <= TextBoxB.Y2 + YDeviation and TextBoxB.Y1 - YDeviation <= TextBoxA.Y2 <= TextBoxB.Y2 + YDeviation):
            VerticalInside = True

        if (
                TextBoxA.X1 - XDeviation <= TextBoxB.X1 <= TextBoxA.X2 + XDeviation and TextBoxA.X1 - XDeviation <= TextBoxB.X2 <= TextBoxA.X2 + XDeviation) or (
                TextBoxB.X1 - XDeviation <= TextBoxA.X1 <= TextBoxB.X2 + XDeviation and TextBoxB.X1 - XDeviation <= TextBoxA.X2 <= TextBoxB.X2 + XDeviation):
            HorizontalInside = True

        if VerticalIntersect or HorizontalIntersect:

            if (TextBoxA.X1 - XDeviation) <= TextBoxB.X1 <= (TextBoxA.X1 + XDeviation) and (VerticalIntersect or VerticalInside):
                Rightbound = True

            if (TextBoxA.X2 + XDeviation) <= TextBoxB.X2 <= (TextBoxA.X2 + XDeviation) and (VerticalIntersect or VerticalInside):
                LeftBound = True

            if Rightbound and LeftBound:
                if TextBoxA.Combined:  # done
                    if TextBoxB.Combined:  # done
                        NewTextBoxesA, IndexListA = ConvertSubPointsToPoints(
                            TextBoxA)  # creates a copy of each origin point to textbox A and returns the new index list
                        NewTextBoxesB, IndexListB = ConvertSubPointsToPoints(TextBoxB,
                                                                             len(IndexListA))  # creates a copy of each origin point to textbox B and returns the new index list
                        TextBoxes[Comb[0]] = SubPointsToSubPoints(NewTextBoxesA, NewTextBoxesB, XDeviation, YDeviation)
                        TextBoxes[Comb[0]].PreviousLeftNode = TextBoxA.Copy()
                        TextBoxes[Comb[0]].PreviousRightNode = TextBoxB.Copy()
                        TextBoxes.pop(Comb[1])
                        return PointComparison(TextBoxes, XMD, YMD)
                    else:  # done
                        NewTextBoxesA, IndexListA = ConvertSubPointsToPoints(TextBoxA)
                        TextBoxes[Comb[0]] = SubPointsToSubPoints(NewTextBoxesA, [TextBoxB], XDeviation, YDeviation)
                        TextBoxes[Comb[0]].PreviousLeftNode = TextBoxA.Copy()
                        TextBoxes[Comb[0]].PreviousRightNode = TextBoxB.Copy()
                        TextBoxes.pop(Comb[1])
                        return PointComparison(TextBoxes, XMD, YMD)
                if TextBoxB.Combined:  # done
                    NewTextBoxesB, IndexListB = ConvertSubPointsToPoints(TextBoxB)
                    TextBoxes[Comb[0]] = SubPointsToSubPoints(NewTextBoxesB, [TextBoxA], XDeviation, YDeviation)
                    TextBoxes[Comb[0]].PreviousLeftNode = TextBoxA.Copy()
                    TextBoxes[Comb[0]].PreviousRightNode = TextBoxB.Copy()
                    TextBoxes.pop(Comb[1])
                    return PointComparison(TextBoxes, YMD, XMD)
                else:
                    X2List = list((TextBoxA.X2, TextBoxB.X2))
                    X1List = list((TextBoxA.X1, TextBoxB.X1))
                    TextBoxes[Comb[0]].PreviousLeftNode = TextBoxA.Copy()
                    TextBoxes[Comb[0]].PreviousRightNode = TextBoxB.Copy()
                    TextBoxes[Comb[0]].X2 = int((sum(X2List) / len(X2List)))
                    TextBoxes[Comb[0]].X1 = int((sum(X1List) / len(X1List)))
                    if (TextBoxA.Y1 - YDeviation) <= TextBoxB.Y2 <= (TextBoxA.Y1 + YDeviation):
                        TextBoxes[Comb[0]].SubNodes.append(
                            Node(TextBoxes[Comb[0]].X1, TextBoxes[Comb[0]].X2, TextBoxB.Y1,
                                 int((TextBoxA.Y1 + TextBoxB.Y2) / 2)))
                        TextBoxes[Comb[0]].SubNodes.append(
                            Node(TextBoxes[Comb[0]].X1, TextBoxes[Comb[0]].X2, int((TextBoxA.Y1 + TextBoxB.Y2) / 2),
                                 TextBoxA.Y2))
                        TextBoxes[Comb[0]].Y1 = TextBoxB.Y1
                        TextBoxes[Comb[0]].Y2 = TextBoxA.Y2
                    if (TextBoxA.Y2 - YDeviation) <= TextBoxB.Y1 <= (TextBoxA.Y2 + YDeviation):
                        TextBoxes[Comb[0]].SubNodes.append(
                            Node(TextBoxes[Comb[0]].X1, TextBoxes[Comb[0]].X2, TextBoxA.Y1,
                                 int((TextBoxA.Y1 + TextBoxB.Y2) / 2)))
                        TextBoxes[Comb[0]].SubNodes.append(
                            Node(TextBoxes[Comb[0]].X1, TextBoxes[Comb[0]].X2, int((TextBoxA.Y1 + TextBoxB.Y2) / 2),
                                 TextBoxB.Y2))
                        TextBoxes[Comb[0]].Y1 = TextBoxA.Y1
                        TextBoxes[Comb[0]].Y2 = TextBoxB.Y2
                    TextBoxes[Comb[0]].Combined = True
                    TextBoxes[Comb[0]].CombinedType = 4
                    TextBoxes.pop(Comb[1])
                    return PointComparison(TextBoxes, XMD, YMD)
            elif Rightbound:
                if TextBoxA.Combined:  # done
                    if TextBoxB.Combined:  # done
                        NewTextBoxesA, IndexListA = ConvertSubPointsToPoints(
                            TextBoxA)  # creates a copy of each origin point to textbox A and returns the new index list
                        NewTextBoxesB, IndexListB = ConvertSubPointsToPoints(TextBoxB,
                                                                             len(IndexListA))  # creates a copy of each origin point to textbox B and returns the new index list
                        TextBoxes[Comb[0]] = SubPointsToSubPoints(NewTextBoxesA, NewTextBoxesB, XDeviation, YDeviation)
                        TextBoxes[Comb[0]].PreviousLeftNode = TextBoxA.Copy()
                        TextBoxes[Comb[0]].PreviousRightNode = TextBoxB.Copy()
                        TextBoxes.pop(Comb[1])
                        return PointComparison(TextBoxes, XMD, YMD)
                    else:  # done
                        NewTextBoxesA, IndexListA = ConvertSubPointsToPoints(TextBoxA)
                        TextBoxes[Comb[0]] = SubPointsToSubPoints(NewTextBoxesA, [TextBoxB], XDeviation, YDeviation)
                        TextBoxes[Comb[0]].PreviousLeftNode = TextBoxA.Copy()
                        TextBoxes[Comb[0]].PreviousRightNode = TextBoxB.Copy()
                        TextBoxes.pop(Comb[1])
                        return PointComparison(TextBoxes, XMD, YMD)
                if TextBoxB.Combined:  # done
                    NewTextBoxesB, IndexListB = ConvertSubPointsToPoints(TextBoxB)
                    TextBoxes[Comb[0]] = SubPointsToSubPoints(NewTextBoxesB, [TextBoxA], XDeviation, YDeviation)
                    TextBoxes[Comb[0]].PreviousLeftNode = TextBoxA.Copy()
                    TextBoxes[Comb[0]].PreviousRightNode = TextBoxB.Copy()
                    TextBoxes.pop(Comb[1])
                    return PointComparison(TextBoxes, YMD, XMD)
                else:
                    X2List = list((TextBoxA.X2, TextBoxB.X2))
                    X1List = list((TextBoxA.X1, TextBoxB.X1))
                    TextBoxes[Comb[0]].PreviousLeftNode = TextBoxA.Copy()
                    TextBoxes[Comb[0]].PreviousRightNode = TextBoxB.Copy()
                    TextBoxes[Comb[0]].X2 = int((sum(X2List) / len(X2List)))
                    TextBoxes[Comb[0]].X1 = min(X1List)
                    if (TextBoxA.Y1 - YDeviation) <= TextBoxB.Y2 <= (TextBoxA.Y1 + YDeviation):
                        TextBoxes[Comb[0]].SubNodes.append(
                            Node(TextBoxB.X1, int(sum(X2List) / len(X2List)), TextBoxB.Y1,
                                 int((TextBoxA.Y1 + TextBoxB.Y2) / 2)))
                        TextBoxes[Comb[0]].SubNodes.append(
                            Node(TextBoxA.X1, int(sum(X2List) / len(X2List)), int((TextBoxA.Y1 + TextBoxB.Y2) / 2),
                                 TextBoxA.Y2))
                        TextBoxes[Comb[0]].Y1 = TextBoxB.Y1
                        TextBoxes[Comb[0]].Y2 = TextBoxA.Y2
                    if (TextBoxA.Y2 - YDeviation) <= TextBoxB.Y1 <= (TextBoxA.Y2 + YDeviation):
                        TextBoxes[Comb[0]].SubNodes.append(
                            Node(TextBoxA.X1, int(sum(X2List) / len(X2List)), TextBoxA.Y1,
                                 int((TextBoxA.Y1 + TextBoxB.Y2) / 2)))
                        TextBoxes[Comb[0]].SubNodes.append(
                            Node(TextBoxB.X1, int(sum(X2List) / len(X2List)), int((TextBoxA.Y1 + TextBoxB.Y2) / 2),
                                 TextBoxB.Y2))
                        TextBoxes[Comb[0]].Y1 = TextBoxA.Y1
                        TextBoxes[Comb[0]].Y2 = TextBoxB.Y2
                    TextBoxes[Comb[0]].Combined = True
                    TextBoxes[Comb[0]].CombinedType = 4
                    TextBoxes.pop(Comb[1])
                    return PointComparison(TextBoxes, XMD, YMD)
            elif LeftBound:
                if TextBoxA.Combined:  # done
                    if TextBoxB.Combined:  # done
                        NewTextBoxesA, IndexListA = ConvertSubPointsToPoints(
                            TextBoxA)  # creates a copy of each origin point to textbox A and returns the new index list
                        NewTextBoxesB, IndexListB = ConvertSubPointsToPoints(TextBoxB,
                                                                             len(IndexListA))  # creates a copy of each origin point to textbox B and returns the new index list
                        TextBoxes[Comb[0]] = SubPointsToSubPoints(NewTextBoxesA, NewTextBoxesB, XDeviation, YDeviation)
                        TextBoxes[Comb[0]].PreviousLeftNode = TextBoxA.Copy()
                        TextBoxes[Comb[0]].PreviousRightNode = TextBoxB.Copy()
                        TextBoxes.pop(Comb[1])
                        return PointComparison(TextBoxes, XMD, YMD)
                    else:  # done
                        NewTextBoxesA, IndexListA = ConvertSubPointsToPoints(TextBoxA)
                        TextBoxes[Comb[0]] = SubPointsToSubPoints(NewTextBoxesA, [TextBoxB], XDeviation, YDeviation)
                        TextBoxes[Comb[0]].PreviousLeftNode = TextBoxA.Copy()
                        TextBoxes[Comb[0]].PreviousRightNode = TextBoxB.Copy()
                        TextBoxes.pop(Comb[1])
                        return PointComparison(TextBoxes, XMD, YMD)
                if TextBoxB.Combined:  # done
                    NewTextBoxesB, IndexListB = ConvertSubPointsToPoints(TextBoxB)
                    TextBoxes[Comb[0]] = SubPointsToSubPoints(NewTextBoxesB, [TextBoxA], XDeviation, YDeviation)
                    TextBoxes[Comb[0]].PreviousLeftNode = TextBoxA.Copy()
                    TextBoxes[Comb[0]].PreviousRightNode = TextBoxB.Copy()
                    TextBoxes.pop(Comb[1])
                    return PointComparison(TextBoxes, YMD, XMD)
                else:
                    X2List = list((TextBoxA.X2, TextBoxB.X2))
                    X1List = list((TextBoxA.X1, TextBoxB.X1))
                    TextBoxes[Comb[0]].PreviousLeftNode = TextBoxA.Copy()
                    TextBoxes[Comb[0]].PreviousRightNode = TextBoxB.Copy()
                    TextBoxes[Comb[0]].X2 = int((sum(X2List) / len(X2List)))
                    TextBoxes[Comb[0]].X1 = min(X1List)
                    if (TextBoxA.Y1 - YDeviation) <= TextBoxB.Y2 <= (TextBoxA.Y1 + YDeviation):
                        TextBoxes[Comb[0]].SubNodes.append(
                            Node(int(sum(X1List) / len(X1List)), TextBoxB.X2, TextBoxB.Y1,
                                 int((TextBoxA.Y1 + TextBoxB.Y2) / 2)))
                        TextBoxes[Comb[0]].SubNodes.append(
                            Node(int(sum(X1List) / len(X1List)), TextBoxA.X2, int((TextBoxA.Y1 + TextBoxB.Y2) / 2),
                                 TextBoxA.Y2))
                        TextBoxes[Comb[0]].Y1 = TextBoxB.Y1
                        TextBoxes[Comb[0]].Y2 = TextBoxA.Y2
                    if (TextBoxA.Y2 - YDeviation) <= TextBoxB.Y1 <= (TextBoxA.Y2 + YDeviation):
                        TextBoxes[Comb[0]].SubNodes.append(
                            Node(int(sum(X1List) / len(X1List)), TextBoxA.X2, TextBoxA.Y1,
                                 int((TextBoxA.Y1 + TextBoxB.Y2) / 2)))
                        TextBoxes[Comb[0]].SubNodes.append(
                            Node(int(sum(X1List) / len(X1List)), TextBoxB.X2, int((TextBoxA.Y1 + TextBoxB.Y2) / 2),
                                 TextBoxB.Y2))
                        TextBoxes[Comb[0]].Y2 = TextBoxB.Y2
                        TextBoxes[Comb[0]].Y1 = TextBoxA.Y1
                    TextBoxes[Comb[0]].Combined = True
                    TextBoxes[Comb[0]].CombinedType = 5
                    TextBoxes.pop(Comb[1])
                    return PointComparison(TextBoxes, XMD, YMD)
    CombListAux = CombListIn.copy()
    for Comb in CombListIn:
        TextBoxA = TextBoxes[Comb[0]]
        TextBoxB = TextBoxes[Comb[1]]
        VerticalInside = False
        VerticalIntersect = False
        HorizontalIntersect = False
        HorizontalInside = False
        Tcase = False
        LShrink = False
        if (
                TextBoxA.Y1 - YDeviation <= TextBoxB.Y1 <= TextBoxA.Y2 + YDeviation and TextBoxB.Y1 - YDeviation <= TextBoxA.Y2 <= TextBoxB.Y2 + YDeviation) or (
                TextBoxB.Y1 - YDeviation <= TextBoxA.Y1 <= TextBoxB.Y2 + YDeviation and TextBoxA.Y1 - YDeviation <= TextBoxB.Y2 <= TextBoxA.Y2 + YDeviation):
            VerticalIntersect = True
        if (
                TextBoxA.X1 - XDeviation <= TextBoxB.X1 <= TextBoxA.X2 + XDeviation and TextBoxB.X1 - XDeviation <= TextBoxA.X2 <= TextBoxB.X2 + XDeviation) or (
                TextBoxB.X1 - XDeviation <= TextBoxA.X1 <= TextBoxB.X2 + XDeviation and TextBoxA.X1 - XDeviation <= TextBoxB.X2 <= TextBoxA.X2 + XDeviation):
            HorizontalIntersect = True
        if (
                TextBoxA.Y1 - YDeviation <= TextBoxB.Y1 <= TextBoxA.Y2 + YDeviation and TextBoxA.Y1 - YDeviation <= TextBoxB.Y2 <= TextBoxA.Y2 + YDeviation) or (
                TextBoxB.Y1 - YDeviation <= TextBoxA.Y1 <= TextBoxB.Y2 + YDeviation and TextBoxB.Y1 - YDeviation <= TextBoxA.Y2 <= TextBoxB.Y2 + YDeviation):
            VerticalInside = True

        if (
                TextBoxA.X1 - XDeviation <= TextBoxB.X1 <= TextBoxA.X2 + XDeviation and TextBoxA.X1 - XDeviation <= TextBoxB.X2 <= TextBoxA.X2 + XDeviation) or (
                TextBoxB.X1 - XDeviation <= TextBoxA.X1 <= TextBoxB.X2 + XDeviation and TextBoxB.X1 - XDeviation <= TextBoxA.X2 <= TextBoxB.X2 + XDeviation):
            HorizontalInside = True

        if VerticalIntersect or HorizontalIntersect:
            if HorizontalIntersect and VerticalInside:
                LShrink = True

            if VerticalIntersect and (HorizontalIntersect or HorizontalInside):
                Tcase = True

            if LShrink:
                if TextBoxA.Combined:  # done
                    if TextBoxB.Combined:  # done
                        NewTextBoxesA, IndexListA = ConvertSubPointsToPoints(
                            TextBoxA)  # creates a copy of each origin point to textbox A and returns the new index list
                        NewTextBoxesB, IndexListB = ConvertSubPointsToPoints(TextBoxB,
                                                                             len(IndexListA))  # creates a copy of each origin point to textbox B and returns the new index list
                        TextBoxes[Comb[0]] = SubPointsToSubPoints(NewTextBoxesA, NewTextBoxesB, XDeviation, YDeviation)
                        TextBoxes[Comb[0]].PreviousLeftNode = TextBoxA.Copy()
                        TextBoxes[Comb[0]].PreviousRightNode = TextBoxB.Copy()
                        TextBoxes.pop(Comb[1])
                        return PointComparison(TextBoxes, XMD, YMD)
                    else:  # done
                        NewTextBoxesA, IndexListA = ConvertSubPointsToPoints(TextBoxA)
                        TextBoxes[Comb[0]] = SubPointsToSubPoints(NewTextBoxesA, [TextBoxB], XDeviation, YDeviation)
                        TextBoxes[Comb[0]].PreviousLeftNode = TextBoxA.Copy()
                        TextBoxes[Comb[0]].PreviousRightNode = TextBoxB.Copy()
                        TextBoxes.pop(Comb[1])
                        return PointComparison(TextBoxes, XMD, YMD)
                if TextBoxB.Combined:  # done
                    NewTextBoxesB, IndexListB = ConvertSubPointsToPoints(TextBoxB)
                    TextBoxes[Comb[0]] = SubPointsToSubPoints(NewTextBoxesB, [TextBoxA], XDeviation, YDeviation)
                    TextBoxes[Comb[0]].PreviousLeftNode = TextBoxA.Copy()
                    TextBoxes[Comb[0]].PreviousRightNode = TextBoxB.Copy()
                    TextBoxes.pop(Comb[1])
                    return PointComparison(TextBoxes, YMD, XMD)
                else:
                    TextBoxes[Comb[0]] = SubPointsToSubPoints([TextBoxB], [TextBoxA], XDeviation, YDeviation)
                    TextBoxes[Comb[0]].PreviousLeftNode = TextBoxA.Copy()
                    TextBoxes[Comb[0]].PreviousRightNode = TextBoxB.Copy()
                    TextBoxes[Comb[0]].Combined = True
                    TextBoxes[Comb[0]].CombinedType = 7
                    TextBoxes.pop(Comb[1])
                    '''
                    X2List = list((TextBoxA.X2, TextBoxB.X2))
                    X1List = list((TextBoxA.X1, TextBoxB.X1))
                    Y1List = list((TextBoxA.Y1, TextBoxB.Y1))
                    Y2List = list((TextBoxA.Y2, TextBoxB.Y2))
                    TextBoxes[Comb[0]].PreviousLeftNode = TextBoxA.Copy()
                    TextBoxes[Comb[0]].PreviousRightNode = TextBoxB.Copy()
                    TextBoxes[Comb[0]].X2 = max(X2List)
                    TextBoxes[Comb[0]].X1 = min(X1List)
                    TextBoxes[Comb[0]].Y2 = max(Y2List)
                    TextBoxes[Comb[0]].Y1 = min(Y1List)
                    if (TextBoxA.Y1 - YDeviation) <= TextBoxB.Y1 <= (TextBoxA.Y1 + YDeviation):
                        TextBoxes[Comb[0]].SubNodes.append(
                            Node(min(X1List), max(X2List), int(sum(Y1List) / len(Y1List)), min(Y2List)))
                        TextBoxes[Comb[0]].SubNodes.append(Node(min(X1List), min(X2List), min(Y2List), max(Y2List)))

                    if (TextBoxA.Y2 - YDeviation) <= TextBoxB.Y2 <= (TextBoxA.Y2 + YDeviation):
                        TextBoxes[Comb[0]].SubNodes.append(Node(min(X1List), min(X2List), min(Y1List), max(Y1List)))
                        TextBoxes[Comb[0]].SubNodes.append(
                            Node(min(X1List), max(X2List), max(Y1List), int(sum(Y2List) / len(Y2List))))
                    else:
                        CombListAux.pop(CombListAux.index(Comb))
                        return PointComparison(TextBoxes, XMD, YMD, CombListIn=CombListAux)

                    TextBoxes[Comb[0]].Combined = True
                    TextBoxes[Comb[0]].CombinedType = 7
                    TextBoxes.pop(Comb[1])'''
                    return PointComparison(TextBoxes, XMD, YMD)
            elif Tcase:
                if TextBoxA.Combined:  # done
                    if TextBoxB.Combined:  # done
                        NewTextBoxesA, IndexListA = ConvertSubPointsToPoints(
                            TextBoxA)  # creates a copy of each origin point to textbox A and returns the new index list
                        NewTextBoxesB, IndexListB = ConvertSubPointsToPoints(TextBoxB,
                                                                             len(IndexListA))  # creates a copy of each origin point to textbox B and returns the new index list
                        TextBoxes[Comb[0]] = SubPointsToSubPoints(NewTextBoxesA, NewTextBoxesB, XDeviation, YDeviation)
                        TextBoxes[Comb[0]].PreviousLeftNode = TextBoxA.Copy()
                        TextBoxes[Comb[0]].PreviousRightNode = TextBoxB.Copy()
                        TextBoxes.pop(Comb[1])
                        return PointComparison(TextBoxes, XMD, YMD)
                    else:  # done
                        NewTextBoxesA, IndexListA = ConvertSubPointsToPoints(TextBoxA)
                        TextBoxes[Comb[0]] = SubPointsToSubPoints(NewTextBoxesA, [TextBoxB], XDeviation, YDeviation)
                        TextBoxes[Comb[0]].PreviousLeftNode = TextBoxA.Copy()
                        TextBoxes[Comb[0]].PreviousRightNode = TextBoxB.Copy()
                        TextBoxes.pop(Comb[1])
                        return PointComparison(TextBoxes, XMD, YMD)
                if TextBoxB.Combined:  # done
                    NewTextBoxesB, IndexListB = ConvertSubPointsToPoints(TextBoxB)
                    TextBoxes[Comb[0]] = SubPointsToSubPoints(NewTextBoxesB, [TextBoxA], XDeviation, YDeviation)
                    TextBoxes[Comb[0]].PreviousLeftNode = TextBoxA.Copy()
                    TextBoxes[Comb[0]].PreviousRightNode = TextBoxB.Copy()
                    TextBoxes.pop(Comb[1])
                    return PointComparison(TextBoxes, YMD, XMD)
                else:
                    TextBoxes[Comb[0]] = SubPointsToSubPoints([TextBoxB], [TextBoxA], XDeviation, YDeviation)
                    TextBoxes[Comb[0]].PreviousLeftNode = TextBoxA.Copy()
                    TextBoxes[Comb[0]].PreviousRightNode = TextBoxB.Copy()
                    TextBoxes[Comb[0]].Combined = True
                    TextBoxes[Comb[0]].CombinedType = 6
                    TextBoxes.pop(Comb[1])
                    '''
                    X2List = list((TextBoxA.X2, TextBoxB.X2))
                    X1List = list((TextBoxA.X1, TextBoxB.X1))
                    Y1List = list((TextBoxA.Y1, TextBoxB.Y1))
                    Y2List = list((TextBoxA.Y2, TextBoxB.Y2))
                    TextBoxes[Comb[0]].PreviousLeftNode = TextBoxA.Copy()
                    TextBoxes[Comb[0]].PreviousRightNode = TextBoxB.Copy()
                    TextBoxes[Comb[0]].X2 = max(X2List)
                    TextBoxes[Comb[0]].X1 = min(X1List)
                    TextBoxes[Comb[0]].Y2 = max(Y2List)
                    TextBoxes[Comb[0]].Y1 = min(Y1List)
                    if (TextBoxA.Y1 - YDeviation) <= TextBoxB.Y2 <= (TextBoxA.Y1 + YDeviation):
                        TextBoxes[Comb[0]].SubNodes.append(
                            Node(TextBoxB.X1, TextBoxB.X2, TextBoxA.Y1, int((TextBoxB.Y2 + TextBoxA.Y1) / 2)))
                        TextBoxes[Comb[0]].SubNodes.append(
                            Node(TextBoxA.X1, TextBoxA.X2, int((TextBoxB.Y2 + TextBoxA.Y1) / 2), TextBoxA.Y1))

                    if (TextBoxA.Y1 - YDeviation) <= TextBoxB.Y1 <= (TextBoxA.Y2 + YDeviation):
                        TextBoxes[Comb[0]].SubNodes.append(
                            Node(TextBoxA.X1, TextBoxA.X2, TextBoxA.Y1, int((TextBoxB.Y2 + TextBoxA.Y1) / 2)))
                        TextBoxes[Comb[0]].SubNodes.append(
                            Node(TextBoxB.X1, TextBoxB.X2, int((TextBoxB.Y2 + TextBoxA.Y1) / 2), TextBoxA.Y1))
                    else:
                        CombListAux.pop(CombListAux.index(Comb))
                        return PointComparison(TextBoxes, XMD, YMD, CombListIn=CombListAux)

                    TextBoxes[Comb[0]].Combined = True
                    TextBoxes[Comb[0]].CombinedType = 6
                    TextBoxes.pop(Comb[1])'''
                    return PointComparison(TextBoxes, XMD, YMD)
        else:
            CombListAux.pop(CombListAux.index(Comb))
    return TextBoxes


@dataclass
class Point:
    Type_class: int
    X1: int
    X2: int
    X1_1: int
    X2_1: int
    datatype = int

    def __init__(self, X1: int, Y1: int ):
        self.X1 = X1
        self.Y1 = Y1
        self.X2 = -1
        self.Y2 = -1
        self.Type_class = -1

    def Type(self) -> str:
        return self.datatype[self.Type_class]

    def edit(self, X2: int, Y2: int, Type_class: int):
        self.X2 = X2
        self.Y2 = Y2
        self.Type_class = Type_class

    def to_String(self):
        print("-------------------------------")
        print("Type_class  ->" + str(self.Type()))
        print("X1  ->" + str(self.X1))
        print("Y1  ->" + str(self.Y1))
        print("X2  ->" + str(self.X2))
        print("Y2  ->" + str(self.Y2))
        print("-------------------------------\n")

    def Print(self):
        i = 0
        datatype = ["Specimen", "Family", "Specimen/Family", "Date", "Location", "Location_detailed", "Ramdom Name",
                    "Ramdom int", "RamdomCharInt", "Other", "Empty"]
        for elem in datatype:
            print(str(i) + "->" + elem)
            i += 1

    def toJSON(self):
        return json.dumps(self, default=lambda o: o.__dict__,
                          sort_keys=True, indent=4)


def on_click(event):
    """states : 0-First point / 1-second point """

    global state
    global image_data
    global Cached_point
    global ax

    if event.button is MouseButton.LEFT:
        try:
            if not state or state == 0:
                print("1? point @ = ")
                print([event.xdata, event.ydata])
                print("\n")
                Cached_point = Point(int(event.xdata), int(event.ydata))
                if event.xdata != None or event.ydata != None:
                    state = 1
            elif state == 1:
                print("2? point @ = ")
                print([event.xdata, event.ydata])
                print("\n")
                #val = int(input(Cached_point.Print()))
                val = int(0)
                while not isinstance(val, int) and 0 <= val <= 10:
                    val = int(0)
                Cached_point.edit(X2=int(event.xdata), Y2=int(event.ydata), Type_class=val)
                image_data["Points"].append(Cached_point)
                rect = Rectangle((Cached_point.X1 if Cached_point.X1 < event.xdata else event.xdata,
                                  Cached_point.Y1 if Cached_point.Y1 < event.ydata else event.ydata),
                                 abs(event.xdata - Cached_point.X1), abs(event.ydata - Cached_point.Y1), linewidth=1,
                                 edgecolor='r', facecolor='none')
                ax.add_patch(rect)
                plt.gcf().canvas.draw()
                if event.xdata != None or event.ydata != None:
                    state = 0
            else:
                print([event.xdata, event.ydata])
        except NameError:
            print("1? point @ = ")
            print([event.xdata, event.ydata])
            print("\n")
            Cached_point = Point(int(event.xdata), int(event.ydata))
            if event.xdata != None or event.ydata != None:
                state = 1
    if event.button is MouseButton.RIGHT:
        if state == 1:
            state = 0
        elif state == 0:
            lastelem = image_data["Points"].pop(-1)
            lastelem.edit(X2=int(event.xdata), Y2=int(event.ydata), Type_class=val)
            image_data["Points"].append(lastelem)
        else:
            print([event.xdata, event.ydata])
    if event.button is MouseButton.MIDDLE:
        if len(image_data["Points"]) > 0:
            Delete_last = int(input("Delete Last = 1 , Other = #"))
            while not isinstance(Delete_last, int):
                Delete_last = int(input("Delete Last = 1 , Other = 2"))
            if Delete_last == 1:
                lastelem = image_data["Points"].pop(-1)
                rect = Rectangle((lastelem.X1, lastelem.Y1), abs(lastelem.X1-lastelem.X2),  abs(lastelem.Y1-lastelem.Y2), linewidth=2, edgecolor='w', facecolor='none')
                ax.add_patch(rect)
                state = 0
            else:
                i = 0
                for elem in image_data["Points"]:
                    print("index  ->" + str(i) + ":")
                    elem.to_String()
                    i += 1
                Delete_last = int(input("Pickapoint"))
                while not isinstance(Delete_last, int) and 0 <= Delete_last < len(image_data["Points"]):
                    Delete_last = int(input("Pickapoint"))
                image_data["Points"].pop(Delete_last-len(image_data["Points"]))
                lastelem = image_data["Points"].pop(-1)
                rect = Rectangle(
                    (lastelem.X1, lastelem.Y1), abs(lastelem.X1 - lastelem.X2), abs(lastelem.Y1 - lastelem.Y2),
                    linewidth=2, edgecolor='w', facecolor='none')
                ax.add_patch(rect)
                state = 0


def on_close(event):
    print('disconnecting callback')
    plt.disconnect(binding_id1)
    plt.disconnect(binding_id2)
    plt.gcf().canvas.stop_event_loop()


def Image_Json_Creator(LabelPath, SkipClustering=True):
    global binding_id1
    global binding_id2
    global image_data
    global ax

    Listing = os.listdir(LabelPath)
    Listing: List[str] = list(filter(lambda elem: '.json' in elem, Listing))
    if Listing:
        Val = str(input('There is already .json created wanna skip step?(Y/N)')).lower().strip()
        print(Val[:1])
        while Val[:1] != "y" and Val[:1] != "n":
            Val = str(input('There is already .json created wanna skip step?(Y/N)')).lower().strip()
            print(Val[:1])
        if Val[0] == "y":
            return
    listing = os.listdir(LabelPath)
    listing: List[str] = list(
        filter(lambda elem: '.jpeg' in elem or '.png' in elem or '.gif' in elem or '.tiff' in elem or '.raw' in elem or '.jpg' in elem,
               listing))
    for filename in listing:
        image_data = {"height": int, "width": int, "path": str, "Points": []}
        img = plt.imread(LabelPath + '/' + filename)
        image_data['height'] = img.shape[0]
        image_data['width'] = img.shape[1]
        image_data['path'] = LabelPath + '/' + filename
        imgplot = plt.imshow(img)
        ax = plt.gca()
        fig = plt.gcf()
        binding_id1 = fig.canvas.mpl_connect('button_press_event', on_click)
        binding_id2 = fig.canvas.mpl_connect('close_event', on_close)
        plt.show(block=True)
        aux = []
        for elem in image_data["Points"]:
            aux.append(elem.__dict__)
        image_data["Points"] = aux
        if not SkipClustering:
            Clustering = AgglomerativeClustering(n_clusters=None, distance_threshold=image_data['width'] * 0.01)
            X = []
            for Point in image_data["Points"]:
                X.append([Point["X1"], -1])
                X.append([Point["X2"], -1])
            Clusters = Clustering.fit(X)
            Clusters_Types = list(set(Clusters.labels_))
            for Clusters_Type in Clusters_Types:
                Clusters_Indices = [j for j, x in enumerate(Clusters.labels_) if x == Clusters_Type]
                Clusters_value = int(sum([X[i][0] for i in Clusters_Indices]) / len(Clusters_Indices))
                for Clusters_i in Clusters_Indices:
                    Point_index = int(Clusters_i / 2)
                    if Clusters_i % 2 == 0:
                        image_data["Points"][Point_index]["X2"] = Clusters_value
                    else:
                        image_data["Points"][Point_index]["X1"] = Clusters_value

            Clustering = AgglomerativeClustering(n_clusters=None, distance_threshold=image_data['height'] * 0.01)
            X = []
            for Point in image_data["Points"]:
                X.append([Point["Y1"], -1])
                X.append([Point["Y2"], -1])
            Clusters = Clustering.fit(X)
            Clusters_Types = list(set(Clusters.labels_))
            for Clusters_Type in Clusters_Types:
                Clusters_Indices = [j for j, x in enumerate(Clusters.labels_) if x == Clusters_Type]
                Clusters_value = int(sum([X[i][0] for i in Clusters_Indices]) / len(Clusters_Indices))
                for Clusters_i in Clusters_Indices:
                    Point_index = int(Clusters_i / 2)
                    if Clusters_i % 2 == 0:
                        image_data["Points"][Point_index]["Y2"] = Clusters_value
                    else:
                        image_data["Points"][Point_index]["Y1"] = Clusters_value
        with open(LabelPath + '/' + filename[:filename.index('.')] + '.json', 'w') as New_json_file:
            json.dump(image_data, New_json_file, ensure_ascii=False)


def TextBoxTextOrganizer(TextBoxPoints, TextToLabel, FontPath, id, LabelText, Size = None , Single_Line_Height_Boost = 1.6):
    Widthslimit = False
    Heightslimit = False
    NewPoint = {'Y2': int, 'Y1': int, 'X2': int, 'X1': int}
    for TextBoxPoint in TextBoxPoints:
        NewPoints = []
        Heights = []
        Widths = []
        if len(TextBoxPoint.SubNodes) > 1:
            LinesAvailable = len(TextBoxPoint.SubNodes)
            for SubNode in TextBoxPoint.SubNodes:
                if SubNode.X2 > SubNode.X1:
                    NewPoint['X1'] = SubNode.X1
                    NewPoint['X2'] = SubNode.X2
                else:
                    NewPoint['X2'] = SubNode.X1
                    NewPoint['X1'] = SubNode.X2
                if SubNode.Y2 > SubNode.Y1:
                    NewPoint['Y1'] = SubNode.Y1
                    NewPoint['Y2'] = SubNode.Y2
                else:
                    NewPoint['Y2'] = SubNode.Y1
                    NewPoint['Y1'] = SubNode.Y2
                NewPoints.append(NewPoint.copy())
                Heights.append(abs(SubNode.Y2 - SubNode.Y1))
                Widths.append(abs(SubNode.X2 - SubNode.X1))
        else:
            LinesAvailable = 1
            if TextBoxPoint.X2 > TextBoxPoint.X1:
                NewPoint['X1'] = TextBoxPoint.X1
                NewPoint['X2'] = TextBoxPoint.X2
            else:
                NewPoint['X2'] = TextBoxPoint.X1
                NewPoint['X1'] = TextBoxPoint.X2
            if TextBoxPoint.Y2 > TextBoxPoint.Y1:
                NewPoint['Y1'] = TextBoxPoint.Y1
                NewPoint['Y2'] = TextBoxPoint.Y2
            else:
                NewPoint['Y2'] = TextBoxPoint.Y1
                NewPoint['Y1'] = TextBoxPoint.Y2
            NewPoints.append(NewPoint.copy())
            Heights = [abs(TextBoxPoint.Y2 - TextBoxPoint.Y1)]
            Widths = [abs(TextBoxPoint.X2 - TextBoxPoint.X1)]

        if Size is None:
            for size in range(1, 100):
                Font = ImageFont.FreeTypeFont(FontPath, size=size)
                left, top, right, bottom = Font.getbbox(TextToLabel)
                w = right - left
                h = bottom - top
                if w > sum(Widths):
                    Widthslimit = True
                    Heightslimit = False
                    break
                if h > min(Heights):
                    Heightslimit = True
                    Widthslimit = False
                    break
                wpre = w
                hpre = h
                SelectedSize = size
            if Widthslimit:
                NewPointToDivide = {'Y2': int, 'Y1': int, 'X2': int, 'X1': int, 'Height': int, 'Split': int, 'AvgHeight': int}
                NewNewPoints = []
                for Pointtemp in NewPoints:
                    NewPointToDivide['Split'] = 1
                    NewPointToDivide['AvgHeight'] = abs(Pointtemp["Y2"] - Pointtemp["Y1"])
                    NewPointToDivide['Height'] = abs(Pointtemp["Y2"] - Pointtemp["Y1"])
                    NewPointToDivide['X1'] = Pointtemp["X1"]
                    NewPointToDivide['X2'] = Pointtemp["X2"]
                    NewPointToDivide['Y1'] = Pointtemp["Y1"]
                    NewPointToDivide['Y2'] = Pointtemp["Y2"]
                    NewNewPoints.append(NewPointToDivide.copy())
                NewPoints = NewNewPoints.copy()
                while Widthslimit:
                    PreviousSelectedSize = SelectedSize
                    AvgHeightList = [aux["AvgHeight"] for aux in NewPoints]
                    IndexOrder = sorted(range(len(AvgHeightList)), key=lambda k: AvgHeightList[k], reverse=True)
                    NewPoints[IndexOrder[0]]['Split'] += 1
                    NewPoints[IndexOrder[0]]['AvgHeight'] = NewPoints[IndexOrder[0]]['Height'] / NewPoints[IndexOrder[0]]['Split']
                    SumVal = 0
                    for Pointtemp in NewPoints:
                        SumVal += ((Pointtemp['X2'] - Pointtemp['X1']) * Pointtemp['Split'])
                    for size in range(1, 100):
                        Font = ImageFont.FreeTypeFont(FontPath, size=size)
                        left, top, right, bottom = Font.getbbox(TextToLabel)
                        w = right - left
                        h = bottom - top
                        if w > SumVal:
                            Widthslimit = True
                            Heightslimit = False
                            break
                        if h > min([aux["AvgHeight"] for aux in NewPoints]):
                            Heightslimit = True
                            Widthslimit = False
                            break
                        SelectedSize = size
                    if SelectedSize > PreviousSelectedSize:
                        wpre = w
                        hpre = h
                    else:
                        NewPoints[IndexOrder[0]]['Split'] -= 1
                        break

                NewPoint = {'Y2': int, 'Y1': int, 'X2': int, 'X1': int}
                NewNewPoints = []
                NewWidths = []
                NewHeights = []
                for Pointtemp in NewPoints:
                    if Pointtemp['Split'] >= 2:
                        NumSplit = Pointtemp['Split']
                        AvgHeight = int((Pointtemp["Y2"] - Pointtemp["Y1"]) / NumSplit)
                        NewPoint['X1'] = Pointtemp["X1"]
                        NewPoint['X2'] = Pointtemp["X2"]
                        NewPoint['Y1'] = Pointtemp["Y1"]
                        NewPoint['Y2'] = (Pointtemp["Y1"] + AvgHeight)
                        while NumSplit >= 1:
                            NewNewPoints.append(NewPoint.copy())
                            NewWidths.append(Pointtemp["X2"] - Pointtemp["X1"])
                            NewHeights.append(AvgHeight)
                            NewPoint['Y1'] += AvgHeight
                            NewPoint['Y2'] += AvgHeight
                            NumSplit -= 1
                    else:
                        NewPoint['X1'] = Pointtemp["X1"]
                        NewPoint['X2'] = Pointtemp["X2"]
                        NewPoint['Y1'] = Pointtemp["Y1"]
                        NewPoint['Y2'] = Pointtemp["Y2"]
                        NewNewPoints.append(NewPoint.copy())
                        NewWidths.append(Pointtemp["X2"] - Pointtemp["X1"])
                        NewHeights.append(Pointtemp["Y2"] - Pointtemp["Y1"])
                Widths = NewWidths.copy()
                Heights = NewHeights.copy()
                NewPoints = NewNewPoints.copy()
                LinesAvailable = len(NewPoints)

            if LinesAvailable > 1:
                EndWidths = 0
                Heightslimit = False
                Widthslimit = True
                while Widthslimit and EndWidths <= len(Widths):
                    for size in range(1, 100):
                        Font = ImageFont.FreeTypeFont(FontPath, size=size)
                        left, top, right, bottom = Font.getbbox(TextToLabel)
                        w = right - left
                        h = bottom - top
                        if w > sum(Widths[0:EndWidths]):
                            Widthslimit = True
                            Heightslimit = False
                            break
                        if h > min(Heights):
                            Heightslimit = True
                            Widthslimit = False
                            break
                        wpre = w
                        hpre = h
                        SelectedSize = size
                    if EndWidths == len(Widths):
                        break
                    else:
                        EndWidths += 1
                if EndWidths == 1:
                    for size in range(1, 100):
                        Font = ImageFont.FreeTypeFont(FontPath, size=size)
                        left, top, right, bottom = Font.getbbox(TextToLabel)
                        w = right - left
                        h = bottom - top
                        if w > Widths[0]:
                            break
                        if h > int(Heights[0] * Single_Line_Height_Boost):
                            break
                        wpre = w
                        hpre = h
                        SelectedSize = size
                    LabelText["X1"].append(NewPoints[0]['X1'])
                    LabelText["X2"].append(NewPoints[0]['X2'])
                    LabelText["Y1"].append(NewPoints[0]['Y1'])
                    LabelText["Y2"].append(NewPoints[0]['Y2'])
                    LabelText["h"].append(hpre)
                    LabelText["Size"].append(SelectedSize - 1)
                    LabelText["Text"].append(TextToLabel)
                    LabelText["id"].append(id)
                else:
                    Begin = 0
                    for ii in range(EndWidths):
                        TextIndexSplit = int((Widths[ii] * len(TextToLabel)) / wpre) + Begin
                        j = 0
                        if TextIndexSplit >= len(TextToLabel):
                            TextIndexSplit = len(TextToLabel)
                            TextSplited = TextToLabel[Begin:TextIndexSplit]
                        else:
                            while TextIndexSplit + j <= len(TextToLabel) and TextIndexSplit - j >= 0 and j <= 3:
                                if (TextToLabel[TextIndexSplit - 1 + j] in " ,-/\\") or TextIndexSplit + j == len(TextToLabel):
                                    TextIndexSplit += j
                                    TextSplited = TextToLabel[Begin:TextIndexSplit]
                                    break
                                elif (TextToLabel[TextIndexSplit - 1 - j] in " ,-/\\") or TextIndexSplit - j == 0:
                                    TextIndexSplit -= j
                                    TextSplited = TextToLabel[Begin:TextIndexSplit]
                                    break
                                elif j == 3:
                                    TextIndexSplit -= 1
                                    TextSplited = TextToLabel[Begin:TextIndexSplit] + "-"
                                    break
                                j += 1

                        Begin = TextIndexSplit
                        LabelText["X1"].append(NewPoints[ii]['X1'])
                        LabelText["X2"].append(NewPoints[ii]['X2'])
                        LabelText["Y1"].append(NewPoints[ii]['Y1'])
                        LabelText["Y2"].append(NewPoints[ii]['Y2'])
                        LabelText["h"].append(hpre)
                        LabelText["Size"].append(SelectedSize - 1)
                        LabelText["Text"].append(TextSplited)
                        LabelText["id"].append(id)
            else:
                for size in range(1, 100): #<--------------------------added this
                    Font = ImageFont.FreeTypeFont(FontPath, size=size)
                    left, top, right, bottom = Font.getbbox(TextToLabel)
                    w = right - left
                    h = bottom - top
                    if w > Widths[0]:
                        break
                    if h > int(Heights[0] * Single_Line_Height_Boost):
                        break
                    wpre = w
                    hpre = h
                    SelectedSize = size
                LabelText["X1"].append(NewPoints[0]['X1'])
                LabelText["X2"].append(NewPoints[0]['X2'])
                LabelText["Y1"].append(NewPoints[0]['Y1'])
                LabelText["Y2"].append(NewPoints[0]['Y2'])
                LabelText["h"].append(hpre)
                LabelText["Size"].append(SelectedSize - 1)
                LabelText["Text"].append(TextToLabel)
                LabelText["id"].append(id)
            id += 1
            return LabelText, id
        else:
            Font = ImageFont.FreeTypeFont(FontPath, size=Size)
            left, top, right, bottom = Font.getbbox(TextToLabel)
            w = right - left
            if w > sum(Widths):
                Widthslimit = True
            else:
                Widthslimit = False

            if Widthslimit:
                NewPointToDivide = {'Y2': int, 'Y1': int, 'X2': int, 'X1': int, 'Height': int, 'Split': int, 'AvgHeight': int}
                NewNewPoints = []
                for Pointtemp in NewPoints:
                    NewPointToDivide['Split'] = 1
                    NewPointToDivide['AvgHeight'] = Pointtemp["Y2"] - Pointtemp["Y1"]
                    NewPointToDivide['Height'] = Pointtemp["Y2"] - Pointtemp["Y1"]
                    NewPointToDivide['X1'] = Pointtemp["X1"]
                    NewPointToDivide['X2'] = Pointtemp["X2"]
                    NewPointToDivide['Y1'] = Pointtemp["Y1"]
                    NewPointToDivide['Y2'] = Pointtemp["Y2"]
                    NewNewPoints.append(NewPointToDivide.copy())
                NewPoints = NewNewPoints.copy()
                while Widthslimit:
                    AvgHeightList = [aux["AvgHeight"] for aux in NewPoints]
                    IndexOrder = sorted(range(len(AvgHeightList)), key=lambda k: AvgHeightList[k])
                    NewPoints[IndexOrder[0]]['Split'] += 1
                    NewPoints[IndexOrder[0]]['AvgHeight'] = NewPoints[IndexOrder[0]]['Height'] / NewPoints[IndexOrder[0]]['Split']
                    SumVal = 0
                    for Pointtemp in NewPoints:
                        SumVal += ((Pointtemp['X2'] - Pointtemp['X1']) * Pointtemp['Split'])
                    if w > sum(Widths):
                        Widthslimit = True
                    else:
                        Widthslimit = False

                NewPoint = {'Y2': int, 'Y1': int, 'X2': int, 'X1': int}
                NewNewPoints = []
                NewWidths = []
                NewHeights = []
                for Pointtemp in NewPoints:
                    if Pointtemp['Split'] >= 2:
                        NumSplit = Pointtemp['Split']
                        AvgHeight = int((abs(Pointtemp["Y2"] - Pointtemp["Y1"]) / NumSplit))
                        NewPoint['X1'] = Pointtemp["X1"]
                        NewPoint['X2'] = Pointtemp["X2"]
                        NewPoint['Y1'] = Pointtemp["Y1"]
                        NewPoint['Y2'] = Pointtemp["Y1"] + AvgHeight
                        while NumSplit >= 1:
                            NewNewPoints.append(NewPoint.copy())
                            NewWidths.append(abs(Pointtemp["X2"] - Pointtemp["X1"]))
                            NewHeights.append(AvgHeight)
                            NewPoint['Y1'] += AvgHeight
                            NewPoint['Y2'] += AvgHeight
                            NumSplit -= 1
                    else:
                        NewPoint['X1'] = Pointtemp["X1"]
                        NewPoint['X2'] = Pointtemp["X2"]
                        NewPoint['Y1'] = Pointtemp["Y1"]
                        NewPoint['Y2'] = Pointtemp["Y2"]
                        NewNewPoints.append(NewPoint.copy())
                        NewWidths.append(abs(Pointtemp["X2"] - Pointtemp["X1"]))
                        NewHeights.append(abs(Pointtemp["Y2"] - Pointtemp["Y1"]))
                Widths = NewWidths.copy()
                Heights = NewHeights.copy()
                NewPoints = NewNewPoints.copy()
                LinesAvailable = len(NewPoints)

            if LinesAvailable > 1:
                Begin = 0
                for ii in range(LinesAvailable):
                    TextIndexSplit = int((Widths[ii] * len(TextToLabel)) / w) + Begin
                    j = 0
                    if TextIndexSplit >= len(TextToLabel):
                        TextIndexSplit = len(TextToLabel)
                        TextSplited = TextToLabel[Begin:TextIndexSplit]
                    else:
                        while TextIndexSplit + j <= len(TextToLabel) and TextIndexSplit - j >= 0 and j <= 3:
                            if (TextToLabel[TextIndexSplit - 1 + j] in " ,-/\\") or TextIndexSplit + j == len(
                                    TextToLabel):
                                TextSplited = TextToLabel[Begin:TextIndexSplit]
                                TextIndexSplit += j
                                break
                            elif (TextToLabel[TextIndexSplit - 1 - j] in " ,-/\\") or TextIndexSplit - j == 0:
                                TextSplited = TextToLabel[Begin:TextIndexSplit]
                                TextIndexSplit -= j
                                break
                            elif j == 3:
                                TextSplited = TextToLabel[Begin:TextIndexSplit - 1] + "-"
                                TextIndexSplit -= 1
                                break
                            j += 1
                    Begin = TextIndexSplit
                    LabelText["X1"].append(NewPoints[ii]['X1'])
                    LabelText["X2"].append(NewPoints[ii]['X2'])
                    LabelText["Y1"].append(NewPoints[ii]['Y1'])
                    LabelText["Y2"].append(NewPoints[ii]['Y2'])
                    LabelText["h"].append(-1)
                    LabelText["Size"].append(Size)
                    LabelText["Text"].append(TextSplited)
                    LabelText["id"].append(id)
            else:
                LabelText["X1"].append(NewPoints[0]['X1'])
                LabelText["X2"].append(NewPoints[0]['X2'])
                LabelText["Y1"].append(NewPoints[0]['Y1'])
                LabelText["Y2"].append(NewPoints[0]['Y2'])
                LabelText["h"].append(-1)
                LabelText["Size"].append(Size)
                LabelText["Text"].append(TextToLabel)
                LabelText["id"].append(id)
            id += 1
            return LabelText, id


size = 512
ImgBlank = np.full([size, size, 3], fill_value=255, dtype=np.uint8)
IM = Image.fromarray(ImgBlank)
IM.save("./Blank.png")
Image_Json_Creator("./", SkipClustering=True)

with open("./Blank.json") as Json_File:
    NewLabelData = json.load(Json_File)
Points = NewLabelData["Points"]
Points.sort(key=lambda x: x["Type_class"], reverse=False)

try:
    os.mkdir("./results")
except OSError as error:
    print('Folder creation error or already exists ')

Point_Types = list(set([elem["Type_class"] for elem in Points]))
OldPoints = []
for Point_Type in Point_Types:
    Indices = [ji for ji, x in enumerate(Points) if x["Type_class"] == Point_Type]
    Nodes = PointsToNodesConversion(Points, Indices)
    OldPoints.append(Nodes.copy())

os.remove("./Blank.png")
draw = ImageDraw.Draw(IM)
for Point_Type in Point_Types:
    for OldPoint in OldPoints[Point_Type]:
        draw.rectangle((OldPoint.X1, OldPoint.Y1, OldPoint.X2, OldPoint.Y2), outline="red")
IM.save("./results/Old.png")

NewPoints = []
for Point_Type in Point_Types:
    Indices = [ji for ji, x in enumerate(Points) if x["Type_class"] == Point_Type]
    Nodes = PointsToNodesConversion(Points, Indices)
    NewPoints.append(PointComparison(Nodes, int((size * 0.01)), int((size * 0.01))))

IM = Image.fromarray(ImgBlank)
draw = ImageDraw.Draw(IM)
for Point_Type in Point_Types:
    for NewPoint in NewPoints[Point_Type]:
        if len(NewPoint.SubNodes) > 1:
            draw.rectangle((NewPoint.X1-1, NewPoint.Y1-1, NewPoint.X2+1, NewPoint.Y2+1), outline="blue")
            for SubNode in NewPoint.SubNodes:
                draw.rectangle((SubNode.X1+1, SubNode.Y1+1, SubNode.X2-1, SubNode.Y2-1), outline="green")
        else:
            draw.rectangle((NewPoint.X1-1, NewPoint.Y1-1, NewPoint.X2+1, NewPoint.Y2+1), outline="blue")
            draw.rectangle((NewPoint.X1+1, NewPoint.Y1+1, NewPoint.X2-1, NewPoint.Y2-1), outline="green")
IM.save("./results/New.png")

TextToLabel = "The quick brown fox jumps over the lazy dog The quick brown fox jumps over the lazy dog The quick brown fox jumps over the lazy dog"
FontPath = "./OpenSans-Regular.ttf"
LabelText = {"Text": [], "Size": [], "X1": [], "X2": [], "Y1": [], "Y2": [], "h": [], "id": [], "OriginalText": ""}
i = 0
for Point_Type in Point_Types:
    for NewPoint in NewPoints[Point_Type]:
        LabelText, id = TextBoxTextOrganizer([NewPoint], TextToLabel, FontPath, i, LabelText)
        i += 1

avgSize = int(sum(LabelText["Size"])/len(LabelText["Size"]))
for jj in range(len(LabelText["Size"])):
    if LabelText["Size"][jj] > avgSize:
        LabelText["Size"][jj] = avgSize
for jj in range(len(LabelText["Text"])):
    Font = ImageFont.FreeTypeFont(FontPath, size=LabelText["Size"][jj])
    draw.text(((LabelText['X1'][jj] + LabelText['X2'][jj]) / 2, (LabelText['Y1'][jj] + LabelText['Y2'][jj]) / 2),
              LabelText["Text"][jj], fill='black', anchor='mm', font=Font)
IM.save("./results/NewText.png")
