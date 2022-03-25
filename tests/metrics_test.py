import metrics
from metrics import Metric
import numpy as np
import pytest



p1 = np.zeros((2, 7, 11))
y1 = np.zeros((2, 7, 11))

y1[1][3][5] = 1
y1[1][2][4] = 1
y1[1][5][1] = 1

p1[1][3][5] = 1
p1[1][2][4] = 1
p1[1][5][1] = 1

p2 = np.zeros((2, 7, 11))
y2 = np.zeros((2, 7, 11))

y2[1][3][5] = 1
y2[1][2][4] = 1
y2[1][5][1] = 1

p2[1][3][5] = 1
p2[1][2][4] = 1


p3 = np.zeros((2, 7, 11))
y3 = np.zeros((2, 7, 11))

y3[1][3][5] = 1
y3[1][2][4] = 1

p3[1][3][5] = 1
p3[1][2][4] = 1
p3[1][5][1] = 1


pb = np.stack((p1, p2, p3))
yb = np.stack((y1, y2, y3))

def test_mean_1():
    ac = Metric("pixel_accuracy")
    ac.measure(p1, y1)
    result = ac.total()
    assert result == 1.

def test_mean_2():
    ac = Metric("pixel_accuracy")
    ac.measure(p2, y2)
    result = ac.total()
    shouldbe = (7 * 11 - 1) / (7 * 11)
    assert result == shouldbe

def test_mean_3():
    ac = Metric("pixel_accuracy")
    ac.measure(p3, y3)
    result = ac.total()
    shouldbe = (7 * 11 - 1) / (7 * 11)
    assert result == shouldbe

def test_mean_4():
    ac = Metric("pixel_accuracy")
    ac.measure(pb, yb)
    result = ac.total()
    shouldbe = (7 * 11 * 3 - 2) / (7 * 11 * 3)
    assert result == shouldbe

def test_mean_5():
    ac = Metric("pixel_accuracy")
    ac.measure(p1, y1)
    ac.measure(p2, y2)
    ac.measure(p3, y3)
    result = ac.total()
    shouldbe = (7 * 11 * 3 - 2) / (7 * 11 * 3)
    assert result == shouldbe

def test_mean_6():
    ac = Metric("pixel_accuracy")
    ac.measure(pb, yb)
    ac.measure(pb, yb)
    result = ac.total()
    shouldbe = (7 * 11 * 6 - 4) / (7 * 11 * 6)
    assert result == shouldbe


########################

def test_precision_1():
    ac = Metric("precision")
    ac.measure(p1, y1)
    result = ac.total()
    assert result == 1.

def test_precision_2():
    ac = Metric("precision")
    ac.measure(p2, y2)
    result = ac.total()
    assert result == 1.

def test_precision_3():
    ac = Metric("precision")
    ac.measure(p3, y3)
    result = ac.total()
    shouldbe = 2/3
    assert result == shouldbe

def test_precision_4():
    ac = Metric("precision")
    ac.measure(pb, yb)
    result = ac.total()
    shouldbe = 8/9
    assert result == shouldbe


def test_precision_5():
    ac = Metric("precision")
    ac.measure(pb, yb)
    ac.measure(p1, y1)
    ac.measure(p2, y2)
    ac.measure(p3, y3)
    result = ac.total()
    shouldbe = 8/9
    assert result == shouldbe

########################

def test_recall_1():
    ac = Metric("recall")
    ac.measure(p1, y1)
    result = ac.total()
    assert result == 1.

def test_recall_2():
    ac = Metric("recall")
    ac.measure(p2, y2)
    result = ac.total()
    shouldbe = 2/3
    assert result == shouldbe

def test_recall_3():
    ac = Metric("recall")
    ac.measure(p3, y3)
    result = ac.total()
    assert result == 1

def test_recall_4():
    ac = Metric("recall")
    ac.measure(pb, yb)
    result = ac.total()
    shouldbe = 8/9
    assert result == shouldbe

def test_recall_5():
    ac = Metric("recall")
    ac.measure(pb, yb)
    ac.measure(p1, y1)
    ac.measure(p2, y2)
    ac.measure(p3, y3)
    result = ac.total()
    shouldbe = 8/9
    shouldbe = pytest.approx(shouldbe)
    assert result == shouldbe

def test_balanced_1():
    ac = Metric("balanced")
    ac.measure(p1, y1)
    result = ac.total()
    shouldbe = 1.
    assert result == shouldbe

def test_balanced_2():
    ac = Metric("balanced")
    ac.measure(p2, y2)
    result = ac.total()
    shouldbe = 5/6.
    shouldbe = pytest.approx(shouldbe)
    assert result == shouldbe

def test_balanced_3():
    ac = Metric("balanced")
    ac.measure(p3, y3)
    result = ac.total()
    shouldbe = 5/6.
    shouldbe = pytest.approx(shouldbe)
    assert result == shouldbe


def test_balanced_3():
    ac = Metric("balanced")
    ac.measure(pb, yb)
    result = ac.total()
    shouldbe = (1 + 2 * 5/6.) / 3
    shouldbe = pytest.approx(shouldbe)
    assert result == shouldbe
