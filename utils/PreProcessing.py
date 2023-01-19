import binascii
import json
import os
import re
import shutil
from dataclasses import dataclass
from itertools import combinations, permutations
from random import randrange
from tkinter import filedialog
from typing import List
import string
import random
import matplotlib.pyplot as plt
import numpy as np
import scipy
import scipy.cluster
import scipy.misc
import skimage.transform
import PIL as PIL
import cv2
from dateutil.parser import *
from color_transfer import color_transfer
from color_matcher import ColorMatcher
from color_matcher.normalizer import Normalizer
from PIL import Image, ImageFont, ImageDraw
from matplotlib.backend_bases import MouseButton
from matplotlib.patches import Rectangle
from skimage import io
from skimage.exposure import match_histograms
from utils.Blender import Blender
from sklearn.cluster import AgglomerativeClustering

'''-------------------------------------------------GlobalVars-------------------------------------------------------'''
binding_id1: int
binding_id2: int
ax: plt.axes
image_data = {"height": [], "width": [], "path": [], "Points": []}
Cached_point = []
state = 0
'''-------------------------------------------------GlobalVars-------------------------------------------------------'''
'''-------------------------------------------------Configs----------------------------------------------------------'''
# Set Lazy to True for hardcoding the necessary paths
Lazy = True
# Rescale images for Yolov4
Rescale = True
# Resolution target for Yolov4 (needs to be multiple of 32x32)For less distorsion keep the H/W near 1.414 A4 sheet
Width = 608
Height = 608
Ratio = Height / Width
# Generates a GreyScale(L) DataSet for Yolov4 instead of RGB
BW_Yolo = True
# This sets a threshold that the image will no longe be streched and will add white bars to target the Ratio set above
LowThreshold = 0.87
TopThreshold = 1.13
# Sets the croping for Image Captioning to a square image(Necessary for Tiled ouput)
Square = True
# Generates a GreyScale(L) DataSet for image captioning instead of RGB
BW_Captioning = False
# Set the Filler to white instead of extra parts of the image
White = False
'''-------------------------------------------------Configs----------------------------------------------------------'''
'''<-------------------------------------------------------------------------------------------------------------NODE'''
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
    Avgheight = [elem[1] - elem[0] for elem in TestLines]
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


'''<-----------------------------------------------------------------------------------------------------------Points'''
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
                val = int(input(Cached_point.Print()))
                while not isinstance(val, int) and 0 <= val <= 10:
                    val = int(input(Cached_point.Print()))
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
            if event.xdata != None or event.ydata != None:
                Cached_point = Point(int(event.xdata), int(event.ydata))
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
    global binding_id1
    global binding_id2
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


DataType = ["Specimen", "Family", "Specimen/Family", "Date", "Location", "Location_detailed", "Ramdom Name",
                    "Ramdom int", "RamdomCharInt", "Other"]
'''<---------------------------------------------------------------------------------------------------------DataType'''
if Lazy:
    JsonPath = 'E:/Users/Halo_/Desktop/thesis/dataset/Json'
    ImagePath = 'E:/Users/Halo_/Desktop/thesis/dataset/images'
    ViaPath = 'E:/Users/Halo_/PycharmProjects/Thesis_test/VIA/VIA_Herbarium_v2(13-03-22).json'
    DataSetPath = 'E:/Users/Halo_/Desktop/thesis/dataset/Dataset_latest_not_not'
else:
    print("Json Folder")
    JsonPath = filedialog.askdirectory(title="Json Folder")
    print("Images Folder")
    ImagePath = filedialog.askdirectory(title="Images Folder")
    print("Via File")
    ViaPath = filedialog.askopenfilename(title="Via File")
    print("DataSet Home Folder")
    DataSetPath = filedialog.askdirectory(title="DataSet Home Folder")


def Retrieve_Number(Str):
    return int(re.search("[0-9]+", Str).group(0))


def Matcher_Renamer(JsonPath, ImagePath):
    MatchedFiles = {'Number': [], 'JsonPath': [], 'ImagePath': []}
    ListingImage = os.listdir(ImagePath)
    ListingImage: List[str] = list(filter(lambda
                                              elem: '.jpeg' in elem or '.png' in elem or '.gif' in elem or '.tiff' in elem or '.raw' in elem or '.jpg' in elem,
                                          ListingImage))
    ListingJson = os.listdir(JsonPath)
    ListingJson: List[str] = list(filter(lambda elem: '.json' in elem, ListingJson))
    for JsonFilename in ListingJson:
        JsonNumber = Retrieve_Number(JsonFilename)
        MatchedFiles['Number'].append(JsonNumber)
        NewJsonFilename = "json_" + str(JsonNumber) + ".json"
        os.rename(JsonPath + "/" + JsonFilename, JsonPath + "/" + NewJsonFilename)
        MatchedFiles['JsonPath'].append(JsonPath + '/' + NewJsonFilename)
        ListingImageNumber: List[str] = list(filter(lambda elem: Retrieve_Number(elem) == JsonNumber, ListingImage))
        if len(ListingImageNumber) == 0 and len(ListingImageNumber) > 1:
            MatchedFiles['ImagePath'].append("-1")
            MatchedFiles['JsonPath'].append("-1")
        else:
            NewImageFilename = "Image_" + str(JsonNumber) + ".jpg"
            os.rename(ImagePath + '/' + ListingImageNumber[0], ImagePath + "/" + NewImageFilename)
            MatchedFiles['ImagePath'].append(ImagePath + '/' + NewImageFilename)
    return MatchedFiles


def NormalizeDate(date, acceptdualdate=False):
    if "|" in date and date.count("|") == 1:
        dates = date.split("|")
    elif "/" in date and date.count("/") == 1:
        dates = date.split("/")
    elif "-" in date and date.count("-") == 1:
        dates = date.split("-")
    elif "\\" in date and date.count("\\") == 1:
        dates = date.split("\\")
    elif " " in date and date.count(" ") == 1:
        dates = date.split(" ")
    else:
        dates = [date]
    Year = []
    Month = []
    Day = []
    try:
        for date in dates:
            Date = parse(date)
            Day.append(Date.day)
            Month.append(Date.month)
            Year.append(Date.year)
    except:
        try:
            DateNumbers = [abs(int(s)) for s in re.findall("[-+]?[.]?[\d]+(?:,\d\d\d)*[\.]?\d*(?:[eE][-+]?\d+)?", date)]
        except:
            return None
        LasttoAppend = ""
        for DateNumber in DateNumbers:
            if len(str(DateNumber)) == 4:
                Year.append(DateNumber)
                LasttoAppend = "Y"
            elif len(
                    str(DateNumber)) <= 2 and DateNumber <= 12 and LasttoAppend == "Y" or LasttoAppend == "D" and len(
                DateNumbers) % 3 == 0:
                Month.append(DateNumber)
                LasttoAppend = "M"
            elif len(str(DateNumber)) <= 2 and DateNumber <= 31 and LasttoAppend == "Y" and len(
                    DateNumbers) % 3 != 0:
                Day.append(DateNumber)
                LasttoAppend = "D"
            elif len(
                    str(DateNumber)) <= 2 and DateNumber <= 31 and LasttoAppend == "M" or LasttoAppend == "" and len(
                DateNumbers) % 3 == 0:
                Day.append(DateNumber)
                LasttoAppend = "D"
            elif len(str(DateNumber)) == 2:
                if DateNumber >= 22:
                    Year.append(DateNumber + 1900)
                    LasttoAppend = "Y"
                else:
                    Year.append(DateNumber + 2000)
                    LasttoAppend = "Y"
        if Month == [] and len(DateNumbers) % 3 != 0:
            DateStr = ["Jan", "Feb", "Mar", "Apr", "May", "Jun", "Jul", "Aug", "Sep", "Oct",
                       "Nov", "Dec"]
            Found = False
            iii = 0
            while not Found and iii <= 11:
                if DateStr[iii].lower() in date.lower():
                    Found = True
                    Month.append(iii + 1)
                    break
                iii += 1
        if len(Month) == len(Day) == len(Year):
            Yeartemp = Year.copy()
            Year = []
            Monthtemp = Month.copy()
            Month = []
            Daytemp = Day.copy()
            Day = []
            try:
                for i in range(len(Month)):
                    Date = parse(str(Daytemp[i]) + "-" + str(Monthtemp[i]) + "-" + str(Yeartemp[i]))
                    Day.append(Date.day)
                    Month.append(Date.month)
                    Year.append(Date.year)
            except:
                return None

    if len(Month) == 0 or len(Day) == 0 or len(Year) == 0 and len(Month) != len(Day) != len(Year):
        return None
    else:
        if len(Month) == 1 or len(Day) == 1 or len(Year) == 1:
            return str(Day[0]) + "-" + str(Month[0]) + "-" + str(Year[0])
        else:
            if (acceptdualdate):
                if len(Month) == 2 or len(Day) == 2 or len(Year) == 2:
                    if Month[0] == Month[1] and Year[0] == Year[1] and (Day[0] == 0 and Day[1] == 31 or Day[0] == 31 and Day[1] == 1):
                        return str(min(Day)) + "-" + str(Month[0]) + "-" + str(Year[0])
                    if Year[0] == Year[1] and (Day[0] == 0 and Day[1] == 31 or Day[0] == 31 and Day[1] == 1) and (Month[0] == 1 and Month[1] == 12 or Month[0] == 12 and Month[1] == 1):
                        return str(min(Day)) + "-" + str(min(Month)) + "-" + str(Year[0])
                    else:
                        return None
                else:
                    return None
            else:
                return None


def Filter_Json(MatchedFiles):
    ExcludedNumbers = []
    FilteredMatchedFiles = {'Number': [], 'JsonPath': [], 'ImagePath': [], 'ScaledImagePath': []}
    if len(MatchedFiles['Number']) == len(MatchedFiles['JsonPath']) == len(MatchedFiles['ImagePath']):
        for i in range(len(MatchedFiles['Number'])):
            NewData = {'Name/Family': [], 'Location': [], 'Date': []}
            DataPath = MatchedFiles['JsonPath'][i]
            if (DataPath == "-1"):
                ExcludedNumbers.append(MatchedFiles['Number'][i])
            else:
                with open(DataPath) as json_file:
                    data = json.load(json_file)
                    data = data["@graph"]
                    data = data[0]
                    skip = False

                    if "dwc:scientificName" in data:
                        NewData['Name/Family'].append(data["dwc:scientificName"])
                    else:
                        skip = True

                    if not skip:
                        if "dwc:family" in data or "dwc:higherClassification" in data:
                            if "dwc:family" in data:
                                NewData['Name/Family'].append(data["dwc:family"])
                            else:
                                NewData['Name/Family'].append(data["dwc:higherClassification"])

                    if not skip:
                        if "dwc:locality" in data:
                            NewData['Location'].append(data["dwc:locality"])
                        else:
                            skip = True

                    if not skip:
                        if "dwc:country" in data or "dwc:decimalLatitude" in data or "dwc:decimalLongitude" in data and "dwc:geodeticDatum" in data:
                            if "dwc:country" in data:
                                NewData['Location'].append(data["dwc:country"])
                            else:
                                NewData['Location'].append('Lat:' + str(data["dwc:decimalLatitude"]) + "; Long:" + str(
                                    data["dwc:decimalLongitude"]) + '; Dantum:' + data["dwc:geodeticDatum"])
                        else:
                            skip = True

                    if not skip:
                        if "dwc:eventDate" in data and len(data["dwc:eventDate"]) >= 6:
                            date = NormalizeDate(data["dwc:eventDate"])
                        elif "dwc:verbatimEventDate" in data and len(data["dwc:verbatimEventDate"]) >= 6 and date is None:
                            date = NormalizeDate(data["dwc:verbatimEventDate"])
                        if date is not None:
                            NewData['Date'].append(date)
                        else:
                            skip = True

                    if not skip:
                        FilteredMatchedFiles['Number'].append(MatchedFiles['Number'][i])
                        FilteredMatchedFiles['ImagePath'].append(MatchedFiles['ImagePath'][i])
                        FilteredMatchedFiles['ScaledImagePath'].append("")
                        HeadTail = os.path.split(MatchedFiles['JsonPath'][i])
                        FilePath = DataSetPath + '/Json_Filtered/' + HeadTail[1]
                        with open(FilePath, 'w') as NewJsonFile:
                            json.dump(NewData, NewJsonFile, ensure_ascii=False)
                        FilteredMatchedFiles['JsonPath'].append(FilePath)
                    else:
                        ExcludedNumbers.append(MatchedFiles['Number'][i])
        return FilteredMatchedFiles, ExcludedNumbers
    else:
        print("The DataSet size per type doesn't match")
        exit(-1)


def LabelToLabelMatching(FilteredMatchedFiles, ViaPath, LabelPath, LowThreshold, TopThreshold): #todo redo
    LabelToRegionMatching = {'FilteredMatchesFilesIndex': [], 'Y2': [], 'Y1': [], 'X2': [], 'X1': [],
                             'ReplacementLabelPath': []}

    Listing = os.listdir(LabelPath)
    Listing: List[str] = list(filter(lambda elem: '.json' in elem, Listing))
    if not Listing:
        print('List is empty plz fill the Path: ' + LabelPath + ' with the respective labels')
        exit(-1)
    with open(ViaPath) as json_file:
        DataVia = json.load(json_file)
        DataVia = DataVia["_via_img_metadata"]
    IndexDictVia = list(DataVia.keys())
    for i in range(len(DataVia)):
        DataViaElem = DataVia[IndexDictVia[i]]
        Index = None
        while Index is None:
            try:
                Index = FilteredMatchedFiles['Number'].index(Retrieve_Number(DataViaElem['filename']))
            except:
                Index = None
                i += 1
                DataViaElem = DataVia[IndexDictVia[i]]

        ImageRegion = DataViaElem['regions']
        for CurrentRegion in ImageRegion:
            ShapeRegion = CurrentRegion['shape_attributes']
            Y2 = int((ShapeRegion['y'] + ShapeRegion['height']))
            Y1 = int(ShapeRegion['y'])
            X2 = int((ShapeRegion['x'] + ShapeRegion['width']))
            X1 = int(ShapeRegion['x'])
            TempRatio = ShapeRegion['height'] / ShapeRegion['width']
            LowThresholdVal = TempRatio * LowThreshold
            TopThresholdVal = TempRatio * TopThreshold
            for filename in Listing:
                with open(LabelPath + '/' + filename) as Json_File:
                    LabelTextBoxes = json.load(Json_File)
                Ratio = LabelTextBoxes["height"] / LabelTextBoxes["width"]
                if LowThresholdVal < Ratio < TopThresholdVal:
                    LabelToRegionMatching['FilteredMatchesFilesIndex'].append(Index)
                    LabelToRegionMatching['Y2'].append(Y2)
                    LabelToRegionMatching['Y1'].append(Y1)
                    LabelToRegionMatching['X2'].append(X2)
                    LabelToRegionMatching['X1'].append(X1)
                    LabelToRegionMatching['ReplacementLabelPath'].append(LabelPath + '/' + filename)
    return LabelToRegionMatching


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
                    IndexOrder = sorted(range(len(AvgHeightList)), key=lambda k: AvgHeightList[k], reverse=True)
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


def IntToRoman(num):
    val = [
        1000, 900, 500, 400,
        100, 90, 50, 40,
        10, 9, 5, 4,
        1
    ]
    syb = [
        "M", "CM", "D", "CD",
        "C", "XC", "L", "XL",
        "X", "IX", "V", "IV",
        "I"
    ]
    roman_num = ''
    i = 0
    while num > 0:
        for _ in range(num // val[i]):
            roman_num += syb[i]
            num -= val[i]
        i += 1
    return roman_num


def MonthToString(num):
    return [["January","Jan."],["February","Feb."],["March","Mar."],["April","Apr."],["May","May"],["June","Jun."],["July","Jul."],["August","Aug."],["September","Sep."],["October","Oct."],["November","Nov."],["December","Dec."]][num-1][randrange(2)]


def MergeLabelToImage(FilteredMatchedFiles, LabelToRegionMatching, BlurZone = 0.015, reverseBlurZone = True,  MaskUnderLetters = False, Transfer_Type = -1 , NormalizeFakeImagesNumber = 10):
    if len(LabelToRegionMatching['FilteredMatchesFilesIndex']) == len(
            LabelToRegionMatching['ReplacementLabelPath']) == len(LabelToRegionMatching['Y2']) == len(
            LabelToRegionMatching['Y1']) == len(LabelToRegionMatching['X2']) == len(LabelToRegionMatching['X1']):
        UniqueFilteredMatchedFilesIndexes = list(set(LabelToRegionMatching['FilteredMatchesFilesIndex']))
        for UniqueFilteredMatchedFilesIndex in UniqueFilteredMatchedFilesIndexes:
            FilteredMatchedFilesIndex = [i for i, x in enumerate(LabelToRegionMatching['FilteredMatchesFilesIndex']) if x == UniqueFilteredMatchedFilesIndex]

            Points = []
            Point = {'X1': int, 'X2': int, 'Y1': int, 'Y2': int}

            for i in FilteredMatchedFilesIndex:
                Point['X1'] = LabelToRegionMatching['X1'][i]
                Point['X2'] = LabelToRegionMatching['X2'][i]
                Point['Y1'] = LabelToRegionMatching['Y1'][i]
                Point['Y2'] = LabelToRegionMatching['Y2'][i]
                Points.append(Point.copy())

            UniquePoints = None
            UniqueFlag = False
            for Point in Points:
                if UniquePoints is None:
                    UniquePoints = []
                    UniquePoints.append(Point.copy())
                else:
                    for UniquePoint in UniquePoints:
                        if Point['X1'] == UniquePoint['X1'] and Point['X2'] == UniquePoint['X2'] and Point['Y1'] == UniquePoint['Y1'] and Point['Y2'] == UniquePoint['Y2']:
                            UniqueFlag=False
                            break
                    if UniqueFlag:
                        UniquePoints.append(Point.copy())
                    UniqueFlag = True

            ArrayIndexPoints = []
            IndexPointslenghts = []

            for UniquePoint in UniquePoints:
                IndexPoints = {'Point': {'X1': int, 'X2': int, 'Y1': int, 'Y2': int}, 'Indexes':[], 'Length':int}
                IndexPoints['Point']['X1'] = UniquePoint['X1']
                IndexPoints['Point']['X2'] = UniquePoint['X2']
                IndexPoints['Point']['Y1'] = UniquePoint['Y1']
                IndexPoints['Point']['Y2'] = UniquePoint['Y2']
                PointIndexing = [i for i, x in enumerate(Points) if x == UniquePoint]
                for TrueIndex in PointIndexing:
                    IndexPoints['Indexes'].append(FilteredMatchedFilesIndex[TrueIndex])
                IndexPoints['Length'] = len(IndexPoints['Indexes'])
                IndexPointslenghts.append(IndexPoints['Length'])
                ArrayIndexPoints.append(IndexPoints.copy())

            Work = np.empty([len(UniquePoints), NormalizeFakeImagesNumber], dtype=int)
            for j in range(len(UniquePoints)):
                if ArrayIndexPoints[j]['Length'] < NormalizeFakeImagesNumber:
                    Size = NormalizeFakeImagesNumber - ArrayIndexPoints[j]['Length']
                    for d in range(Size):
                        ArrayIndexPoints[j]['Indexes'].append(ArrayIndexPoints[j]['Indexes'][randrange(ArrayIndexPoints[j]['Length'])])
                else:
                    ArrayIndexPoints[j]['Indexes'] = ArrayIndexPoints[j]['Indexes'][0:NormalizeFakeImagesNumber]
                    ArrayIndexPoints[j]['Length'] = NormalizeFakeImagesNumber

                ArrayIndexPoints[j]['Length'] = len(ArrayIndexPoints[j]['Indexes'])
                Work[j,:] = ArrayIndexPoints[j]['Indexes']

            JsonPath = FilteredMatchedFiles['JsonPath'][UniqueFilteredMatchedFilesIndex]
            with open(JsonPath) as Json_File:
                TextData = json.load(Json_File)

            Date = parse(TextData['Date'][0])
            DateTransformMethod = randrange(5)
            delim = ["-", "/"][randrange(2)]
            if randrange(2) == 0:
                if DateTransformMethod == 0:
                    date_text = str(Date.day) + delim + str(Date.month) + delim + str(Date.year)
                elif DateTransformMethod == 1:
                    date_text = IntToRoman(Date.day) + delim + str(Date.month) + delim + str(Date.year)
                elif DateTransformMethod == 2:
                    date_text = str(Date.day) + delim + MonthToString(Date.month) + delim + str(Date.year)
                elif DateTransformMethod == 3:
                    date_text = IntToRoman(Date.day) + delim + MonthToString(Date.month) + delim + str(Date.year)
                elif DateTransformMethod == 4:
                    date_text = str(Date.day) + delim + IntToRoman(Date.month) + delim + str(Date.year)
            else:
                if DateTransformMethod == 0:
                    date_text = str(Date.year) + delim + str(Date.month) + delim + str(Date.day)
                elif DateTransformMethod == 1:
                    date_text = str(Date.year) + delim + str(Date.month) + delim + IntToRoman(Date.day)
                elif DateTransformMethod == 2:
                    date_text = str(Date.year) + delim + MonthToString(Date.month) + delim + str(Date.day)
                elif DateTransformMethod == 3:
                    date_text = str(Date.year) + delim + MonthToString(Date.month) + delim + IntToRoman(Date.day)
                elif DateTransformMethod == 4:
                    date_text = str(Date.year) + delim + IntToRoman(Date.month) + delim + str(Date.day)


            JsonPath, JsonFile = os.path.split(FilteredMatchedFiles['JsonPath'][UniqueFilteredMatchedFilesIndex])

            for i in range(NormalizeFakeImagesNumber):
                ImPath, ImFile = os.path.split(FilteredMatchedFiles['ImagePath'][UniqueFilteredMatchedFilesIndex])
                ImOriginal = Image.open(FilteredMatchedFiles['ImagePath'][UniqueFilteredMatchedFilesIndex])
                Number = Retrieve_Number(ImFile)
                with open(JsonPath + "/json_" + str(Number) + "_fake_" + str(i) + ".json", 'w') as NewJsonFile:
                    json.dump(TextData, NewJsonFile, ensure_ascii=False)

                for j in range(len(UniquePoints)):
                    with open(LabelToRegionMatching['ReplacementLabelPath'][Work[j][i]]) as Json_File:
                        NewLabelData = json.load(Json_File)
                    ImLabel = Image.open(NewLabelData["path"])
                    ImLabelArray = np.array(ImLabel)
                    if ImLabelArray.shape[2] == 4:
                        ImLabelArray = ImLabelArray[:, :, 0:3]
                    ImLabel = Image.fromarray(ImLabelArray)
                    draw = ImageDraw.Draw(ImLabel)
                    HeightOfLabelOnOriginal = int(abs(LabelToRegionMatching['Y1'][Work[j][i]] - LabelToRegionMatching['Y2'][Work[j][i]]))
                    WidthOfLabelOnOriginal = int(abs(LabelToRegionMatching['X1'][Work[j][i]] - LabelToRegionMatching['X2'][Work[j][i]]))
                    LabelText = {"Text": [], "Size": [], "X1": [], "X2": [], "Y1": [], "Y2": [], "h": [], "id": [], "OriginalText": ""}
                    FontPath = FontPathsList[randrange(len(FontPathsList))]
                    TextBoxPoints = []
                    TextToLabel = []
                    Point_TypeList = []
                    Points = NewLabelData["Points"]
                    Points.sort(key=lambda x: x["Type_class"], reverse=False)

                    Clustering = AgglomerativeClustering(n_clusters=None, distance_threshold=NewLabelData["width"] * 0.01)
                    X = []
                    for Point in Points:
                        X.append([Point["X1"], -1])
                        X.append([Point["X2"], -1])
                    Clusters = Clustering.fit(X)
                    Clusters_Types = list(set(Clusters.labels_))
                    for Clusters_Type in Clusters_Types:
                        Clusters_Indices = [j for j, x in enumerate(Clusters.labels_) if x == Clusters_Type]
                        Clusters_value = int(sum([X[i][0] for i in Clusters_Indices])/len(Clusters_Indices))
                        for Clusters_i in Clusters_Indices:
                            Point_index = int(Clusters_i / 2)
                            if Clusters_i % 2 == 0:
                                Points[Point_index]["X2"] = Clusters_value
                            else:
                                Points[Point_index]["X1"] = Clusters_value

                    Clustering = AgglomerativeClustering(n_clusters=None, distance_threshold=NewLabelData["height"] * 0.01)
                    X = []
                    for Point in Points:
                        X.append([Point["Y1"], -1])
                        X.append([Point["Y2"], -1])
                    Clusters = Clustering.fit(X)
                    Clusters_Types = list(set(Clusters.labels_))
                    for Clusters_Type in Clusters_Types:
                        Clusters_Indices = [j for j, x in enumerate(Clusters.labels_) if x == Clusters_Type]
                        Clusters_value = int(sum([X[i][0] for i in Clusters_Indices]) / len(Clusters_Indices))
                        for Clusters_i in Clusters_Indices:
                            Point_index = int(Clusters_i/2)
                            if Clusters_i % 2 == 0:
                                Points[Point_index]["Y2"] = Clusters_value
                            else:
                                Points[Point_index]["Y1"] = Clusters_value

                    Point_Types = list(set([elem["Type_class"] for elem in Points]))
                    try:
                        Point_Types.pop(Point_Types.index(10))
                        Point_Types = [10]+Point_Types
                    except:
                        FavColour = get_dominant_color(ImLabel, 9, fast=False)

                    for Point_Type in Point_Types:
                        Indices = [ji for ji, x in enumerate(Points) if x["Type_class"] == Point_Type]
                        Nodes = PointsToNodesConversion(Points, Indices)
                        TextBoxPoints.append(PointComparison(Nodes, int((NewLabelData["width"] * 0.01)), int((NewLabelData["height"] * 0.01))))
                        Point_TypeList.append(Point_Type)
                        if Point_Type == 0:
                            TextToLabel.append(TextData["Name/Family"][0])
                        elif Point_Type == 1:
                            try:
                                TextToLabel.append(TextData["Name/Family"][1])
                            except:
                                TextToLabel.append("")
                        elif Point_Type == 2:
                            try:
                                TextToLabel.append(TextData["Name/Family"][0] + ", " + TextData["Name/Family"][1])
                            except:
                                TextToLabel.append(TextData["Name/Family"][0])
                        elif Point_Type == 3:
                            TextToLabel.append(date_text)
                        elif Point_Type == 4:
                            TextToLabel.append(TextData["Location"][0])
                        elif Point_Type == 5:
                            try:
                                TextToLabel.append(TextData["Location"][1] + ", " + TextData["Location"][0])
                            except:
                                TextToLabel.append(TextData["Location"][0])
                        elif Point_Type == 6:
                            TextToLabel.append(''.join(random.choice(string.ascii_uppercase) for _ in range(8)))
                        elif Point_Type == 7:
                            TextToLabel.append(''.join(random.choice(string.digits) for _ in range(8)))
                        elif Point_Type == 8:
                            TextToLabel.append(''.join(random.choice(string.ascii_uppercase + string.digits) for _ in range(8)))
                        elif Point_Type == 9:
                            TextToLabel.append(''.join(random.choice(string.ascii_uppercase + string.digits) for _ in range(8)))
                        elif Point_Type == 10:
                            pass
                        else:
                            print('Error in MergeLabelToImage Label Type not expected')
                            exit(-1)

                    for NumTextBoxes in range(len(TextBoxPoints)):
                        id = 0
                        if MaskUnderLetters:
                            for TextBoxPoint in TextBoxPoints[NumTextBoxes]:
                                if len(TextBoxPoint.SubNodes) > 0:
                                    for SubNode in TextBoxPoint.SubNodes:
                                        FavColour = get_dominant_color(ImLabel, 9, fast=False)
                                        draw.rectangle((SubNode.X1, SubNode.Y1, SubNode.X2, SubNode.Y2), fill="#" + FavColour[1])
                                else:
                                    FavColour = get_dominant_color(ImLabel, 9, fast=False)
                                    draw.rectangle((TextBoxPoint.X1, TextBoxPoint.Y1, TextBoxPoint.X2, TextBoxPoint.Y2),
                                                   fill="#" + FavColour[1])
                        else:
                            if Point_TypeList[NumTextBoxes] == 10:
                                for TextBoxPoint in TextBoxPoints[NumTextBoxes]:
                                    if len(TextBoxPoint.SubNodes) > 0:
                                        for SubNode in TextBoxPoint.SubNodes:
                                            FavColour = get_dominant_color(ImLabel, 9, fast=False)
                                            draw.rectangle((SubNode.X1, SubNode.Y1, SubNode.X2, SubNode.Y2), fill="#" + FavColour[1])
                                    else:
                                        FavColour = get_dominant_color(ImLabel, 9, fast=False)
                                        draw.rectangle((TextBoxPoint.X1, TextBoxPoint.Y1, TextBoxPoint.X2, TextBoxPoint.Y2), fill="#" + FavColour[1])
                        if Point_TypeList[NumTextBoxes] != 10:
                            LabelText, id = TextBoxTextOrganizer(TextBoxPoints[NumTextBoxes], TextToLabel[NumTextBoxes], FontPath, id, LabelText)
                    # uncomment this lines and comment the next for to normalize Font size to the minimum size available
                    #Min_size = min(LabelText["Size"])
                    #LabelText, id = TextBoxTextOrganizer(TextBoxPoints[NumTextBoxes], TextToLabel[NumTextBoxes], FontPath, id, LabelText, Min_size)
                    #for jj in range(len(LabelText["Text"])):
                        #Font = ImageFont.FreeTypeFont(FontPath, size=Min_size)
                        #draw.text(((LabelText['X1'][jj] + LabelText['X2'][jj]) / 2, (LabelText['Y1'][jj] + LabelText['Y2'][jj]) / 2), LabelText["Text"][jj], fill='black', anchor='mm', font=Font)
                    avgSize = int(sum(LabelText["Size"])/len(LabelText["Size"]))
                    for jj in range(len(LabelText["Size"])):
                        if LabelText["Size"][jj] > avgSize:
                            LabelText["Size"][jj] = avgSize
                    for jj in range(len(LabelText["Text"])):
                        Font = ImageFont.FreeTypeFont(FontPath, size=LabelText["Size"][jj])
                        draw.text(((LabelText['X1'][jj] + LabelText['X2'][jj]) / 2, (LabelText['Y1'][jj] + LabelText['Y2'][jj]) / 2), LabelText["Text"][jj], fill='black', anchor='mm', font=Font)

                    ImLabel = ImLabel.resize((WidthOfLabelOnOriginal, HeightOfLabelOnOriginal), PIL.Image.ANTIALIAS)
                    X1 = LabelToRegionMatching['X1'][Work[j][i]]
                    X2 = LabelToRegionMatching['X2'][Work[j][i]]
                    Y1 = LabelToRegionMatching['Y1'][Work[j][i]]
                    Y2 = LabelToRegionMatching['Y2'][Work[j][i]]
                    X11 = X1 - int((X2 - X1) * BlurZone)
                    X21 = X2 + int((X2 - X1) * BlurZone)
                    Y11 = Y1 - int((Y2 - Y1) * BlurZone)
                    Y21 = Y2 + int((Y2 - Y1) * BlurZone)
                    ImCrop = ImOriginal.crop((X1, Y1, X2, Y2))
                    ImOriginalArray = np.array(ImCrop)
                    drawOriginal = ImageDraw.Draw(ImOriginal)
                    ImCropOriginal = ImOriginal.crop((X11, Y11, X21, Y21))
                    FavColourOriginal = get_dominant_color(ImCropOriginal, 9, fast=True)
                    drawOriginal.rectangle((X11, Y11, X21, Y21), fill="#" + FavColourOriginal[1])
                    ImOriginal = Blender(ImOriginal, ImCropOriginal, X11, Y11, X21, Y21, "#" + FavColourOriginal[1], 0.005, reverse=not(reverseBlurZone))
                    ImLabelArray = np.array(ImLabel)
                    ImLabelArray = ImLabelArray[..., ::-1].copy()
                    ImOriginalArray = ImOriginalArray[..., ::-1].copy()
                    if Transfer_Type == 0:
                        ImLabelArray = color_transfer(ImOriginalArray, ImLabelArray)
                        ImLabelArray = ImLabelArray[..., ::-1].copy()
                        ImLabel = Image.fromarray(ImLabelArray)
                        if j == len(UniquePoints)-1:
                            Blender(ImOriginal, ImLabel, X1, Y1, X2, Y2, Color="#"+FavColour[1], BlurZone=BlurZone,
                                    path=FakeImagesFilteredPath + "/" + str(Number) + "_fake_" + str(i) + ".png",
                                    reverse=reverseBlurZone)
                        else:
                            ImOriginal = Blender(ImOriginal, ImLabel, X1, Y1, X2, Y2, Color="#"+FavColour[1], BlurZone=BlurZone, reverse=reverseBlurZone)
                    elif Transfer_Type == 1:
                        ImLabelArray = MatchColours(ImOriginalArray, ImLabelArray)
                        ImLabelArray = ImLabelArray[..., ::-1].copy()
                        ImLabel = Image.fromarray(ImLabelArray)
                        if j == len(UniquePoints) - 1:
                            Blender(ImOriginal, ImLabel, X1, Y1, X2, Y2, Color="#"+FavColour[1], BlurZone=BlurZone,
                                    path=FakeImagesFilteredPath + "/" + str(Number) + "_fake_" + str(i) + ".png", reverse=reverseBlurZone)
                        else:
                            ImOriginal = Blender(ImOriginal, ImLabel, X1, Y1, X2, Y2, Color="#"+FavColour[1], BlurZone=BlurZone, reverse=reverseBlurZone)
                    elif Transfer_Type == 2:
                        obj = ColorMatcher(src=ImLabelArray, ref=ImOriginalArray, method='mkl')
                        img_res = obj.main()
                        ImLabelArray = Normalizer(img_res).uint8_norm()
                        ImLabel = Image.fromarray(ImLabelArray)
                        if j == len(UniquePoints) - 1:
                            Blender(ImOriginal, ImLabel, X1, Y1, X2, Y2, Color="#"+FavColour[1], BlurZone=BlurZone,
                                    path=FakeImagesFilteredPath + "/" + str(Number) + "_fake_" + str(i) + ".png",
                                    reverse=reverseBlurZone)
                        else:
                            ImOriginal = Blender(ImOriginal, ImLabel, X1, Y1, X2, Y2, Color="#"+FavColour[1], BlurZone=BlurZone, reverse=reverseBlurZone)
                    else:
                        ImOriginalArray = np.array(ImOriginal)
                        mask = np.zeros((ImOriginalArray.shape[0], ImOriginalArray.shape[1], 1), dtype='float32')
                        for ximg in range(X1, X2):
                            for yimg in range(Y1, Y2):
                                if sum(ImLabelArray[yimg-Y1, ximg-X1]) < 510:
                                    mask[yimg][ximg] = 1
                                    mask[yimg - 1][ximg] = 1
                                    mask[yimg + 1][ximg] = 1
                                    mask[yimg][ximg - 1] = 1
                                    mask[yimg][ximg + 1] = 1
                                    mask[yimg - 1][ximg - 1] = 1
                                    mask[yimg + 1][ximg + 1] = 1
                                    mask[yimg - 1][ximg + 1] = 1
                                    mask[yimg + 1][ximg - 1] = 1
                                    mask[yimg - 2][ximg] = 1
                                    mask[yimg + 2][ximg] = 1
                                    mask[yimg][ximg - 2] = 1
                                    mask[yimg][ximg + 2] = 1
                                    mask[yimg - 2][ximg - 2] = 1
                                    mask[yimg + 2][ximg + 2] = 1
                                    mask[yimg - 2][ximg + 2] = 1
                                    mask[yimg + 2][ximg - 2] = 1
                                    mask[yimg - 3][ximg] = 1
                                    mask[yimg + 3][ximg] = 1
                                    mask[yimg][ximg - 3] = 1
                                    mask[yimg][ximg + 3] = 1
                                    mask[yimg - 3][ximg - 3] = 1
                                    mask[yimg + 3][ximg + 3] = 1
                                    mask[yimg - 3][ximg + 3] = 1
                                    mask[yimg + 3][ximg - 3] = 1
                        for ximg in range(X1, X2):
                            for yimg in range(Y1, Y2):
                                if mask[yimg][ximg] == 1:
                                    ImOriginalArray[yimg][ximg] = ImLabelArray[yimg - Y1, ximg - X1]
                        ImOriginal = Image.fromarray(ImOriginalArray)
                        if j == len(UniquePoints)-1:
                            ImOriginal.save(FakeImagesFilteredPath + "/" + str(Number) + "_fake_" + str(i) + ".png")
                FilteredMatchedFiles['Number'].append(Number)
                FilteredMatchedFiles['JsonPath'].append(JsonPath + "/json_" + str(Number) + "_fake_" + str(i) + ".json")
                FilteredMatchedFiles['ImagePath'].append(FakeImagesFilteredPath + "/" + str(Number) + "_fake_" + str(i) + ".png")
                FilteredMatchedFiles['ScaledImagePath'].append("")
        return FilteredMatchedFiles
    else:
        print(
            'The funcion LabelToLabelMatching failed to create the variable LabelToRegionMatching with the same lengh')
        exit(-1)


def Filter_Format_Images(FilteredMatchedFiles, LowThreshold, TopThreshold, Ratio, KeepRaw=True, BW=False):
    TransformationINFO = {"Image_number": [], "Image_Path": [], "leftfill": [], "rightfill": [], "topfill": [],
                          "botfill": [],
                          "transform_width": [], "transform_height": [], "original_width": [],
                          "original_height": [],
                          "target_width": -1, "ratio": -1}
    LowThresholdVal = Ratio * LowThreshold
    TopThresholdVal = Ratio * TopThreshold
    TransformationINFO['ratio'] = Ratio
    TransformationINFO['target_width'] = Width
    if len(FilteredMatchedFiles['Number']) == len(FilteredMatchedFiles['JsonPath']) == len(FilteredMatchedFiles['ImagePath']) == len(FilteredMatchedFiles['ScaledImagePath']):
        UniqueNumbers = list(set(FilteredMatchedFiles['Number']))
        for UniqueNumber in UniqueNumbers:
            indices = [i for i, x in enumerate(FilteredMatchedFiles['Number']) if x == UniqueNumber]
            for i in indices:
                DataPath = FilteredMatchedFiles['ImagePath'][i]
                if KeepRaw:
                    NewDataPath = DataSetPath + '/Images_Filtered/'
                    shutil.copy(DataPath, NewDataPath)
                    FilteredMatchedFiles['ImagePath'][i] = DataSetPath + '/Images_Filtered/' + os.path.split(FilteredMatchedFiles['ImagePath'][i])[1]
                else:
                    FilteredMatchedFiles['ImagePath'][i] = DataPath
                Im = io.imread(DataPath)
                TransformationINFO['Image_number'].append(FilteredMatchedFiles['Number'][i])
                TransformationINFO['original_height'].append(Im.shape[0])
                TransformationINFO['original_width'].append(Im.shape[1])
                if LowThresholdVal < Im.shape[0] / Im.shape[1] < TopThresholdVal:
                    leftfill = 0
                    rightfill = 0
                    topfill = 0
                    botfill = 0
                    Im = Image.fromarray(Im)
                    TransformationINFO['transform_width'].append(0)
                    TransformationINFO['transform_height'].append(0)
                else:
                    if Im.shape[0] / Im.shape[1] > TopThresholdVal:
                        tofill = ((Im.shape[0] / Ratio) - Im.shape[1]) / 2
                        if tofill - int(tofill) > 0:
                            leftfill = (int(tofill)) + 1
                            rightfill = int(tofill)
                            topfill = 0
                            botfill = 0
                        else:
                            leftfill = int(tofill)
                            rightfill = int(tofill)
                            topfill = 0
                            botfill = 0
                        ImageBlank = np.zeros([Im.shape[0], int(Im.shape[0] / Ratio), 3], dtype=np.uint8)
                        ImageBlank.fill(255)
                        ImageBlank[:, leftfill: leftfill + Im.shape[1]] = Im[:]
                        Im = Image.fromarray(ImageBlank)
                        TransformationINFO['transform_height'].append(ImageBlank.shape[0])
                        TransformationINFO['transform_width'].append(ImageBlank.shape[1])
                    elif Im.shape[0] / Im.shape[1] < LowThresholdVal:
                        tofill = ((Im.shape[1] * Ratio) - Im.shape[0]) / 2
                        if tofill - int(tofill) > 0:
                            topfill = (int(tofill)) + 1
                            botfill = int(tofill)
                            leftfill = 0
                            rightfill = 0
                        else:
                            topfill = int(tofill)
                            botfill = int(tofill)
                            leftfill = 0
                            rightfill = 0
                        ImageBlank = np.zeros([int(Im.shape[1] * Ratio), Im.shape[1], 3], dtype=np.uint8)
                        ImageBlank.fill(255)
                        ImageBlank[topfill: topfill + Im.shape[0], :] = Im[:]
                        Im = Image.fromarray(ImageBlank)
                        TransformationINFO['transform_height'].append(ImageBlank.shape[0])
                        TransformationINFO['transform_width'].append(ImageBlank.shape[1])
                    else:
                        print('Failed to format image')
                        exit(-1)
                TransformationINFO['leftfill'].append(leftfill)
                TransformationINFO['rightfill'].append(rightfill)
                TransformationINFO['topfill'].append(topfill)
                TransformationINFO['botfill'].append(botfill)
                ImageFormatedResized = Im.resize((Width, Height), Image.ANTIALIAS)
                if BW:
                    ImageFormatedResized = ImageFormatedResized.convert('L')
                ResizedFormatedImagePath = DataSetPath + '/Images_Filtered_Scaled/' + os.path.split(FilteredMatchedFiles['ImagePath'][i])[1]
                FilteredMatchedFiles['ScaledImagePath'][i] = ResizedFormatedImagePath
                TransformationINFO['Image_Path'].append(ResizedFormatedImagePath)
                ImageFormatedResized.save(ResizedFormatedImagePath)
                i += 1
        return TransformationINFO
    else:
        print("The DataSet size per type doesn't match")
        exit(-1)


def Scale_Down(ViaPath, TransformationINFO, FilteredMatchedFiles):
    ScaledBoundingBoxes = {"Image_number": [], "Image_path": [], "Y_1": [], "Y_2": [], "X_1": [], "X_2": [],
                           "deformation_X": [],
                           "deformation_Y": []}
    BoundingBoxes = {"Image_number": [], "Image_path": [], "Y_1": [], "Y_2": [], "X_1": [], "X_2": [],
                     "deformation_X": [],
                     "deformation_Y": []}

    with open(ViaPath) as json_file:
        DataVia = json.load(json_file)
        DataVia = DataVia["_via_img_metadata"]
    IndexDictVia = list(DataVia.keys())
    TargetWidth = TransformationINFO['target_width']
    Ratio = TransformationINFO['ratio']
    TargetHeight = int(TargetWidth * Ratio)
    for i in range(len(DataVia)):
        DataViaElem = DataVia[IndexDictVia[i]]
        try:
            Indexes = [i for i, x in enumerate(TransformationINFO['Image_number']) if x == Retrieve_Number(DataViaElem['filename'])]
        except:
            i += 1
            DataViaElem = DataVia[IndexDictVia[i]]
            Indexes = [i for i, x in enumerate(TransformationINFO['Image_number']) if x == Retrieve_Number(DataViaElem['filename'])]
        for Index in Indexes:
            ImageRegion = DataViaElem['regions']
            if TransformationINFO['transform_width'][Index] != 0:
                DeformationY = TransformationINFO['transform_height'][Index] / TargetHeight
            else:
                DeformationY = TransformationINFO['original_height'][Index] / TargetHeight

            if TransformationINFO['transform_height'][Index] != 0:
                DeformationX = TransformationINFO['transform_width'][Index] / TargetWidth
            else:
                DeformationX = TransformationINFO['original_width'][Index] / TargetWidth
            OffsetX = TransformationINFO['leftfill'][Index]
            OffsetY = TransformationINFO['topfill'][Index]
            if len(ImageRegion) != 0:
                ScaledBoundingBoxes["Image_number"].append(TransformationINFO['Image_number'][Index])
                ScaledBoundingBoxes["Image_path"].append(TransformationINFO['Image_Path'][Index])
                ScaledBoundingBoxes["deformation_X"].append(DeformationX)
                ScaledBoundingBoxes["deformation_Y"].append(DeformationY)
                BoundingBoxes["Image_number"].append(TransformationINFO['Image_number'][Index])
                BoundingBoxes["Image_path"].append(FilteredMatchedFiles['ImagePath'][FilteredMatchedFiles['Number'].index(TransformationINFO['Image_number'][Index])])
                BoundingBoxes["deformation_X"].append(1)
                BoundingBoxes["deformation_Y"].append(1)
                TempX1 = []
                TempX2 = []
                TempY1 = []
                TempY2 = []
                for CurrentRegion in ImageRegion:
                    ShapeRegion = CurrentRegion['shape_attributes']
                    TempY2.append(int((ShapeRegion['y'] + OffsetY + ShapeRegion['height']) / DeformationY))
                    TempY1.append(int((ShapeRegion['y'] + OffsetY) / DeformationY))
                    TempX2.append(int((ShapeRegion['x'] + OffsetX + ShapeRegion['width']) / DeformationX))
                    TempX1.append(int((ShapeRegion['x'] + OffsetX) / DeformationX))
                    BoundingBoxes["Y_2"].append(ShapeRegion['y'] + ShapeRegion['height'])
                    BoundingBoxes["Y_1"].append(ShapeRegion['y'])
                    BoundingBoxes["X_2"].append(ShapeRegion['x'] + ShapeRegion['width'])
                    BoundingBoxes["X_1"].append(ShapeRegion['x'])
                    ScaledBoundingBoxes["Y_2"].append(TempY2)
                    ScaledBoundingBoxes["Y_1"].append(TempY1)
                    ScaledBoundingBoxes["X_2"].append(TempX2)
                    ScaledBoundingBoxes["X_1"].append(TempX1)
    return ScaledBoundingBoxes, BoundingBoxes


def Dataset_Yolo(ScaledBoundingBoxes, TransformationINFO=None):
    ID = 0
    Text = ""
    if TransformationINFO is not None:
        Text2 = ""
    Image_id = 0
    for i in range(len(ScaledBoundingBoxes["Image_number"])):
        shutil.copy(ScaledBoundingBoxes["Image_path"][i], YoloDataSetPath + "/image_"+str(Image_id) + ScaledBoundingBoxes["Image_path"][i][-4:])
        X1 = ScaledBoundingBoxes["X_1"][i]
        X2 = ScaledBoundingBoxes["X_2"][i]
        Y1 = ScaledBoundingBoxes["Y_1"][i]
        Y2 = ScaledBoundingBoxes["Y_2"][i]
        Image_Width = TransformationINFO["original_width"][i]
        Image_Height = TransformationINFO["original_height"][i]
        if len(X1) == len(X2) == len(Y1) == len(Y2):
            Line = YoloDataSetPath + "/image_" + str(Image_id) + ScaledBoundingBoxes["Image_path"][i][-4:] + " "
            if TransformationINFO is not None:
                Line2 = YoloDataSetPath + "/image_" + str(Image_id) + ScaledBoundingBoxes["Image_path"][i][-4:] + " "
            for j in range(len(X1)):
                Line += str(X1[j]) + "," + str(Y1[j]) + "," + str(X2[j]) + "," + str(Y2[j]) + "," + str(ID) + " "
                if TransformationINFO is not None:
                    Width = X2[j]-X1[j]
                    Height = Y2[j]-Y1[j]
                    if Height % 2 == 0:
                        Height += 1
                    if Width % 2 == 0:
                        Width += 1
                    DIFF = abs((Height - Width) / 2)
                    DIFF = [DIFF, DIFF]
                    if DIFF[0] - int(DIFF[0]) > 0:
                        DIFF[0] = int(DIFF[0]) + 1
                        DIFF[1] = int(DIFF[1])
                    else:
                        DIFF[0] = int(DIFF[0])
                        DIFF[1] = int(DIFF[1])
                    if Height > Width:
                        if X1[j] - DIFF[0] < 0:
                            DIFF[1] = DIFF[1] + (DIFF[0] - X1[j])
                            DIFF[0] = X1[j]
                        if (X1[j] + Width) + DIFF[1] > Image_Width:
                            DIFF[0] = DIFF[0] + (X1[j] + Width + DIFF[1] - Image_Width)
                            DIFF[1] = DIFF[1] - (X1[j] + Width + DIFF[1] - Image_Width)

                        X_1 = X1[j] - DIFF[0]
                        Y_1 = Y1[j]
                        X_2 = X1[j] + Width + DIFF[1]
                        Y_2 = Y1[j] + Height

                    else:
                        if Y1[j] - DIFF[0] < 0:
                            DIFF[1] = DIFF[1] + (DIFF[0] - Y1[j])
                            DIFF[0] = Y1[j]
                        if (Y1[j] + Height) + DIFF[1] > Image_Height:
                            DIFF[0] = DIFF[0] + (Y1[j] + Height + DIFF[1] - Image_Height)
                            DIFF[1] = DIFF[1] - (Y1[j] + Height + DIFF[1] - Image_Height)

                        X_1 = X1[j]
                        Y_1 = Y1[j] - DIFF[0]
                        X_2 = X1[j] + Width
                        Y_2 = Y1[j] + Height + DIFF[1]

                    Line2 += str(X_1) + "," + str(Y_1) + "," + str(X_2) + "," + str(Y_2) + "," + str(ID) + " "
            Line = Line[:len(Line) - 1] + "\n"
            Text += Line
            if TransformationINFO is not None:
                Line2 = Line[:len(Line) - 1] + "\n"
                Text2 += Line2

        else:
            print("error point size doesnt match")
        Image_id += 1

    TextFile = open(YoloDataSetPath + '/train.txt', "w")
    TextFile.write(Text)
    TextFile.close()

    TextFile = open(YoloDataSetPath + '/train_SQ.txt', "w")
    TextFile.write(Text2)
    TextFile.close()

    TextFile = open(YoloDataSetPath + '/Class.names', "w")
    TextFile.write("meta_data\n")
    TextFile.close()


def Crop(Square, Imagem, X, Y, Width, Height, BW, White=True):
    if Square:
        if not White:
            if Height % 2 == 0:
                Height += 1

            if Width % 2 == 0:
                Width += 1

            DIFF = abs((Height - Width) / 2)
            DIFF = [DIFF, DIFF]
            if DIFF[0] - int(DIFF[0]) > 0:
                DIFF[0] = int(DIFF[0]) + 1
                DIFF[1] = int(DIFF[1])
            else:
                DIFF[0] = int(DIFF[0])
                DIFF[1] = int(DIFF[1])

            if Height > Width:
                if X - DIFF[0] < 0:
                    DIFF[1] = DIFF[1] + (DIFF[0] - X)
                    DIFF[0] = X

                if (X + Width) + DIFF[1] > Imagem.shape[1]:
                    DIFF[0] = DIFF[0] + (X + Width + DIFF[1] - Imagem.shape[1])
                    DIFF[1] = DIFF[1] - (X + Width + DIFF[1] - Imagem.shape[1])

                if Y < 0:
                    Y = 0

                if Y + Height > Imagem.shape[0]:
                    Height = Imagem.shape[0] - Y

                return Imagem[Y:(Y + Height), (X - DIFF[0]):(X + Width + DIFF[1])]
            else:
                if Y - DIFF[0] < 0:
                    DIFF[1] = DIFF[1] + (DIFF[0] - Y)
                    DIFF[0] = Y

                if (Y + Height) + DIFF[1] > Imagem.shape[0]:
                    DIFF[0] = DIFF[0] + (Y + Height + DIFF[1] - Imagem.shape[0])
                    DIFF[1] = DIFF[1] - (Y + Height + DIFF[1] - Imagem.shape[0])

                if X < 0:
                    X = 0

                if X + Width > Imagem.shape[1]:
                    Width = Imagem.shape[1] - X

                return Imagem[(Y - DIFF[0]):(Y + Height + DIFF[1]), X:(X + Width)]
        else:
            if Height % 2 == 0:
                Height += 1

            if Width % 2 == 0:
                Width += 1

            if Height > Width:
                if BW:
                    ImgBlank = np.zeros([Height, Height, 1], dtype=np.uint8)
                else:
                    ImgBlank = np.zeros([Height, Height, 3], dtype=np.uint8)
            else:
                if BW:
                    ImgBlank = np.zeros([Width, Width, 1], dtype=np.uint8)
                else:
                    ImgBlank = np.zeros([Width, Width, 3], dtype=np.uint8)

            ImgBlank.fill(255)

            ImgBlank[int((ImgBlank.shape[1] - Height) / 2):int((ImgBlank.shape[1] - Height) / 2) + Height,
            int((ImgBlank.shape[0] - Width) / 2):int((ImgBlank.shape[0] - Width) / 2) + Width] = Imagem[Y:(Y + Height),
                                                                                                 X:(X + Width)]
            return ImgBlank
    else:
        return Imagem[Y:(Y + Height), X:(X + Width)]


def DataSet_Transformer(JsonViaIndividualPath, ImagesDatasetPath, TestingDataFilteredPath, ImagesFilteredCroppedPath, Square, White, BW, FilteredMatchedFiles, ViaPath):
    XData_New_Training_Crop = {"FullImage": [], "Crop": []}
    YData_New_Training_Crop = {"Number": [], "Name/Family": [], "Location": [], "Date": []}
    XData_New_Validation = {"FullImage": [], "Crop": []}
    YData_New_Validation = {"Number": [], "Name/Family": [], "Location": [], "Date": []}
    Used_Indexes = []

    with open(ViaPath) as json_file:
        DataVia = json.load(json_file)
        DataVia = DataVia["_via_img_metadata"]
    IndexDictVia = list(DataVia.keys())

    for i in range(len(DataVia)):
        DataViaElem = DataVia[IndexDictVia[i]]
        ImageRegion = DataViaElem['regions']
        if len(ImageRegion) != 0:
            ImageNumber = Retrieve_Number(IndexDictVia[i])
            try:
                Indexes = [i for i, x in enumerate(FilteredMatchedFiles['Number']) if x == ImageNumber]
            except:
                i += 1
                ImageNumber = Retrieve_Number(IndexDictVia[i])
                DataViaElem = DataVia[IndexDictVia[i]]
                Indexes = [i for i, x in enumerate(FilteredMatchedFiles['Number']) if x == ImageNumber]
            for Index in Indexes:
                Used_Indexes.append(Index)
                ImageCropGrouped = []
                shutil.copy(FilteredMatchedFiles["ImagePath"][Index], ImagesDatasetPath)
                Imagem = io.imread(ImagesDatasetPath + "/" + os.path.split(FilteredMatchedFiles["ImagePath"][Index])[1])
                if BW:
                    Imagem = Imagem.convert('L')
                with open(FilteredMatchedFiles["JsonPath"][Index]) as Json_File:
                    DataJson = json.load(Json_File)
                JoinedFillOnceName = True
                JoinedFillOnceLocation = True
                JoinedFillOnceDate = True
                j = 1
                Crop_Temp_New_Training = []
                for CurrentRegion in ImageRegion:
                    ShapeRegion = CurrentRegion['shape_attributes']
                    Attributes = CurrentRegion['region_attributes']
                    if 'Yes' in Attributes["Label_Complete"].keys() or "fake" in FilteredMatchedFiles["ImagePath"][Index]:
                        if JoinedFillOnceName:
                            JoinedFillOnceName = False
                        if JoinedFillOnceLocation:
                            JoinedFillOnceLocation = False
                        if JoinedFillOnceDate:
                            JoinedFillOnceDate = False
                    else:
                        if 'Yes' in Attributes["Name/Family"].keys():
                            if JoinedFillOnceName:
                                JoinedFillOnceName = False

                        if 'Yes' in Attributes["Location"].keys():
                            if JoinedFillOnceLocation:
                                JoinedFillOnceLocation = False

                        if 'Yes' in Attributes["Date"].keys():
                            if JoinedFillOnceDate:
                                JoinedFillOnceDate = False

                    ImageCrop = Crop(Square, Imagem, ShapeRegion['x'], ShapeRegion['y'], ShapeRegion['width'],
                                     ShapeRegion['height'], BW, White)
                    Subs = os.path.split(FilteredMatchedFiles["ImagePath"][Index])[1].split('.')
                    NewCropImagePath = ImagesFilteredCroppedPath + "/" + Subs[0] + '_crop_' + str(j) + '.' + Subs[1]
                    Crop_Temp_New_Training.append(NewCropImagePath)
                    plt.imsave(NewCropImagePath, ImageCrop)
                    ImageCropGrouped.append(ImageCrop)
                    j += 1
                if not JoinedFillOnceName and not JoinedFillOnceLocation and not JoinedFillOnceDate:
                    XData_New_Training_Crop["Crop"].append(Crop_Temp_New_Training.copy())
                    XData_New_Training_Crop["FullImage"].append(ImagesDatasetPath + "/" + os.path.split(FilteredMatchedFiles["ImagePath"][Index])[1])
                    YData_New_Training_Crop["Name/Family"].append(DataJson["Name/Family"])
                    YData_New_Training_Crop["Location"].append(DataJson["Location"])
                    YData_New_Training_Crop["Date"].append(DataJson["Date"])
                    YData_New_Training_Crop["Number"].append(ImageNumber)
                Subs = os.path.split(FilteredMatchedFiles["JsonPath"][Index])[1].split('.')
                JsonViaAttributesPath = JsonViaIndividualPath + "/" + Subs[0] + '_Via_Attributes_.' + Subs[1]
                with open(JsonViaAttributesPath, 'w') as New_json_file:
                    json.dump(DataViaElem, New_json_file, ensure_ascii=False)
        i += 1

    for Index in range(len(FilteredMatchedFiles['Number'])):
        if Index not in Used_Indexes:
            with open(FilteredMatchedFiles["JsonPath"][Index]) as Json_File:
                DataJson = json.load(Json_File)
            XData_New_Validation["Crop"].append([])
            XData_New_Validation["FullImage"].append(TestingDataFilteredPath + "/" + os.path.split(FilteredMatchedFiles["ImagePath"][Index])[1])
            YData_New_Validation["Name/Family"].append(DataJson["Name/Family"])
            YData_New_Validation["Location"].append(DataJson["Location"])
            YData_New_Validation["Date"].append(DataJson["Date"])
            YData_New_Validation["Number"].append(FilteredMatchedFiles["Number"][Index])
            shutil.copy(FilteredMatchedFiles["ImagePath"][Index], TestingDataFilteredPath)

    with open(TestingDataFilteredPath + '/Dataset.json', 'w') as New_json_file:
        json.dump({"YData": YData_New_Validation, "XPath": XData_New_Validation}, New_json_file, ensure_ascii=False)

    with open(ImagesFilteredCroppedPath + '/Dataset.json', 'w') as New_json_file:
        json.dump({"YData": YData_New_Training_Crop, "XPath": XData_New_Training_Crop}, New_json_file,
                  ensure_ascii=False)

    YDataFake = {"Number": [], "Name/Family": [], "Location": [], "Date": []}
    XDataFake = {"FullImage": [], "Crop": []}
    FakeIndexes = [i for i, x in enumerate(XData_New_Training_Crop["FullImage"]) if "fake" in x]
    for Fakeindex in FakeIndexes:
        XDataFake["Crop"].append(XData_New_Training_Crop["Crop"][Fakeindex])
        XDataFake["FullImage"].append(XData_New_Training_Crop["FullImage"][Fakeindex])
        YDataFake["Number"].append(YData_New_Training_Crop["Number"][Fakeindex])
        YDataFake["Name/Family"].append(YData_New_Training_Crop["Name/Family"][Fakeindex])
        YDataFake["Location"].append(YData_New_Training_Crop["Location"][Fakeindex])
        YDataFake["Date"].append(YData_New_Training_Crop["Date"][Fakeindex])
    with open(ImagesFilteredCroppedPath + '/FakeDataset.json', 'w') as New_json_file:
        json.dump({"YData": YDataFake, "XPath": XDataFake}, New_json_file, ensure_ascii=False)

    YDataSmall_Validation = {"Number": [], "Name/Family": [], "Location": [], "Date": []}
    XDataSmall_Validation = {"FullImage": [], "Crop": []}
    try:
        SmallIndexes = random.sample(range(0, len(YData_New_Validation["Number"])), len(YData_New_Training_Crop["Number"]))
    except:
        SmallIndexes = random.sample(range(0, len(YData_New_Validation["Number"])), int(len(YData_New_Validation["Number"])/2))
    for Smallindex in SmallIndexes:
        XDataSmall_Validation["Crop"].append([])
        XDataSmall_Validation["FullImage"].append(XData_New_Validation["FullImage"][Smallindex])
        YDataSmall_Validation["Number"].append(YData_New_Validation["Number"][Smallindex])
        YDataSmall_Validation["Name/Family"].append(YData_New_Validation["Name/Family"][Smallindex])
        YDataSmall_Validation["Location"].append(YData_New_Validation["Location"][Smallindex])
        YDataSmall_Validation["Date"].append(YData_New_Validation["Date"][Smallindex])
        shutil.copy(XData_New_Validation["FullImage"][Smallindex], TestingDataFilteredPath+"/Small")
    with open(TestingDataFilteredPath + '/Small/SmallDataset.json', 'w') as New_json_file:
        json.dump({"YData": YDataSmall_Validation, "XPath": XDataSmall_Validation}, New_json_file, ensure_ascii=False)


def ListFonts(FontPath):
    listing = os.listdir(FontPath)
    listing: List[str] = list(filter(lambda elem: '.ttf' in elem or '.otf' in elem, listing))
    return [FontPath + "/" + i for i in listing]


def get_dominant_color(pil_img, NUM_CLUSTERS, fast=False): #TODO IndexError: tuple index out of range
    im = pil_img.copy()
    if fast:
        im = im.resize((150, 150))  # optional, to reduce time
    ar = np.asarray(im)
    shape = ar.shape
    ar = ar.reshape(np.product(shape[:2]), shape[2]).astype(float)
    codes, dist = scipy.cluster.vq.kmeans(ar, NUM_CLUSTERS)
    vecs, dist = scipy.cluster.vq.vq(ar, codes)  # assign codes
    counts, bins = np.histogram(vecs, len(codes))  # count occurrences
    index_max = np.argmax(counts)  # find most frequent
    peak = codes[index_max]
    colour = binascii.hexlify(bytearray(int(c) for c in peak)).decode('ascii')
    return peak, colour


def MatchColours(original, target):
    matched = match_histograms(target, original, multichannel=True)
    return matched


def rgb_to_hex(rgb_color):
    hex_color = "#"
    for i in rgb_color:
        i = int(i)
        hex_color += ("{:02x}".format(i))
    return hex_color

try:
    os.mkdir(DataSetPath)
except OSError as error:
    print('Folder creation error or already exists ')

MatchedFiles = Matcher_Renamer(JsonPath, ImagePath)

with open(DataSetPath + '/MatchedFiles.json', 'w') as New_json_file:
    json.dump(MatchedFiles, New_json_file, ensure_ascii=False)

ImagesFilteredPath = DataSetPath + '/Images_Filtered'
try:
    os.mkdir(ImagesFilteredPath)
except OSError as error:
    print('Folder creation error or already exists ')

ImagesFilteredScaledPath = DataSetPath + '/Images_Filtered_Scaled'
try:
    os.mkdir(ImagesFilteredScaledPath)
except OSError as error:
    print('Folder creation error or already exists ')

JsonFilteredPath = DataSetPath + '/Json_Filtered'
try:
    os.mkdir(JsonFilteredPath)
except OSError as error:
    print('Folder creation error or already exists ')

LabelPath = DataSetPath + '/Labels'
try:
    os.mkdir(LabelPath)
except OSError as error:
    print('Folder creation error or already exists ')
Ignore = input("Now Copy the your labels images to the following path:" + LabelPath + " - - - - (Press Enter to continue...)")
Image_Json_Creator(LabelPath)

FontPath = DataSetPath + '/Fonts'
try:
    os.mkdir(FontPath)
except OSError as error:
    print('Folder creation error or already exists ')
Ignore = input("Now Copy the font files (.ttf) to the following path:" + FontPath + " - - - - (Press Enter to continue...)")
FontPathsList = ListFonts(FontPath)

FakeImagesFilteredPath = DataSetPath + '/Fake_Images_Filtered'
try:
    os.mkdir(FakeImagesFilteredPath)
except OSError as error:
    print('Folder creation error or already exists ')

FilteredMatchedFiles, ExcludedNumbers = Filter_Json(MatchedFiles)
with open(DataSetPath + '/ExcludedNumbers.json', 'w') as New_json_file:
    json.dump(ExcludedNumbers, New_json_file, ensure_ascii=False)

LabelToRegionMatching = LabelToLabelMatching(FilteredMatchedFiles, ViaPath, LabelPath, LowThreshold, TopThreshold)
with open(DataSetPath + '/LabelRegionMatching.json', 'w') as New_json_file:
    json.dump(LabelToRegionMatching, New_json_file, ensure_ascii=False)

FilteredMatchedFiles = MergeLabelToImage(FilteredMatchedFiles, LabelToRegionMatching, Transfer_Type=0, NormalizeFakeImagesNumber=5)

TransformationINFO = Filter_Format_Images(FilteredMatchedFiles, LowThreshold, TopThreshold, Ratio,KeepRaw=False, BW=BW_Yolo)
with open(DataSetPath + '/Images_Filtered_Scaled' + '/ScaleTransformationInfo.json', 'w') as New_json_file:
    json.dump(TransformationINFO, New_json_file, ensure_ascii=False)

ScaledBoundingBoxes, BoundingBoxes = Scale_Down(ViaPath, TransformationINFO, FilteredMatchedFiles)

with open(DataSetPath + '/FilteredMatchedFiles.json', 'w') as New_json_file:
    json.dump(FilteredMatchedFiles, New_json_file, ensure_ascii=False)

with open(DataSetPath + '/Images_Filtered_Scaled' + '/ScaledBoundingBoxes.json', 'w') as New_json_file:
    json.dump(ScaledBoundingBoxes, New_json_file, ensure_ascii=False)

with open(DataSetPath + '/Images_Filtered' + '/BoundingBoxes.json', 'w') as New_json_file:
    json.dump(BoundingBoxes, New_json_file, ensure_ascii=False)

YoloDataSetPath = DataSetPath + '/Yolo_DataSet(' + str(Width) + "x" + str(Height) + ")"
try:
    os.mkdir(YoloDataSetPath)
except OSError as error:
    print('Folder creation error or already exists ')

TransformerDataSet = DataSetPath + '/Transformer_DataSet'
try:
    os.mkdir(TransformerDataSet)
except OSError as error:
    print('Folder creation error or already exists ')

Dataset_Yolo(ScaledBoundingBoxes, TransformationINFO)

ImagesFilteredCroppedPath = TransformerDataSet + '/Images_Filtered_Cropped'
try:
    os.mkdir(ImagesFilteredCroppedPath)
except OSError as error:
    print('Folder creation error or already exists ')

ImagesDatasetPath = TransformerDataSet + '/Images_Filtered'
try:
    os.mkdir(ImagesDatasetPath)
except OSError as error:
    print('Folder creation error or already exists ')

JsonViaIndividualPath = TransformerDataSet + '/Json_Via_Attributes_Individual'
try:
    os.mkdir(JsonViaIndividualPath)
except OSError as error:
    print('Folder creation error or already exists ')

TestingDataFilteredPath = DataSetPath + '/TestingData'
try:
    os.mkdir(TestingDataFilteredPath)
    os.mkdir(TestingDataFilteredPath + '/Small')
except OSError as error:
    print('Folder creation error or already exists ')


DataSet_Transformer(JsonViaIndividualPath, ImagesDatasetPath, TestingDataFilteredPath, ImagesFilteredCroppedPath, Square, White, BW_Captioning, FilteredMatchedFiles, ViaPath)

print("Done"*50)
