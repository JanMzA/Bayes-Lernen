import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go

def bayes_theorem(data):
    """
    P(h|D) = (P(D|h)*P(h)) / P(D)
    P(h) - Wahrscheinlichkeit das h gültig ist
    P(D) - Wahrscheinlichkeit, dass D als Ereignisdatensatz auftritt
    h - Hypothse
    D - Datensatz
    :return: P(h|D) (a posteriori Wahrscheinlichkeit)
    """
    # P(D) = P(D|h) * P(h) + P(D|!h) * P(!h)
    P_D = data["P(D|h)"] * data["P(h)"] + data["P(D|!h)"] * data["P(!h)"]

    return (data["P(D|h)"] * data["P(h)"]) / P_D



def maximum_a_posteriori_hypothese(hypotheses,data):
    """
    h_map = argmax_aller hypothesen P(h|D) * P(h)
    :param hypotheses: Liste aller Hypothesen
    :return: h_map
    """
    probabilities = []

    for hypo in hypotheses:
        prob = data[hypo[0]] * data[hypo[1]]
        probabilities.append(prob)


    max_index = np.argmax(probabilities)
    return hypotheses[max_index]


def konzeptlernen(train_data,hypotheses):
    """
    ges. c: x->{0,1}
    c - Zielhypothese
    Abbildung des Datensatz(x) auf False(0) order True(1) (Bsp. Wir wollen etwas machen oder nicht)
    :param data: x_i mit gegebenem c(x_i)
    :return:
    """
    p_h_gegeben_D = []
    for hypo in hypotheses.index:
        for idx in train_data.index:
            if train_data.at[idx,"Tennis?"] == hypotheses.at[hypo,"Tennis?"]:
                train_data.at[idx,"P_D_gegeben_h"] = 1
            else:
                train_data.at[idx,"P_D_gegeben_h"] = 0

        p_h_gegeben_D.append(1/sum(df["P_D_gegeben_h"]))

    return p_h_gegeben_D


def print_data(data):
    t = np.linspace(-4, 4, 200)



    fig = px.scatter(data,"x","y")
    fig.add_trace(
        go.Scatter(
            x=[-4,4],
            y=[-4,4],
            mode="lines",
            showlegend=False
        )
    )
    fig.add_trace(
        go.Scatter(
            x = t,
            y = t*t,
            showlegend=False
        )
    )
    fig.show()

def functions(type,value):

    if type == "1*x":
        return value
    elif type =="x*x":
        return value * value


def funktionslernern(data,funktions_hypothesen):
    """
    gesucht wird eine reell wertige Zielfunktion
    gegeben sind die Punkte x_i und d_i
    d_i = f(x_i) + e_i
    e_i - normalverteilte zufallsvaribale

    h_ml = argmin( sum(d_i - h(x_i))^2)
    :param data:
    :return:
    """

    #Messfehler hinzufügen
    data["error"] = np.random.randn(len(data))
    data["d_i"] = data["y"] + data["error"]
    # target values bestimmen
    for hypo in funktions_hypothesen:
        for idx in data.index:
            erg = functions(hypo,data.at[idx,"x"])
            data.at[idx,hypo] = erg

    fehlerquadrate = []

    for hypothese in funktions_hypothesen:
        sum  = 0
        for idx in data.index:
            erg = (data.at[idx,"d_i"] - data.at[idx,hypothese]) **2
            sum += erg

        fehlerquadrate.append(erg)


    h_ML_idx = np.argmin(fehlerquadrate)
    h_ML = funktions_hypothesen[h_ML_idx]

    print_data(data)

    return h_ML






# h-Krebs, D- test positiv
data = {"P(h)": 0.008, "P(!h)": 0.992, "P(D|h)": 0.98,
        "P(!D|h)": 0.02, "P(D|!h)": 0.03, "P(!D|!h)": 0.97}


# Beobachtung neuer Patient, Test positiv. Hat der neue Patient Krebs?
hypotheses = [["P(D|h)","P(h)"],["P(D|!h)","P(!h)"]]




# möglich für Konzept lernen?
df = pd.DataFrame({
    "Vorhersage": ["sonnig", "sonnig", "bedeckt", "regnerisch", "regnerisch", "regnerisch",
                   "bedeckt", "sonnig", "sonnig", "regnerisch", "sonnig", "bedeckt",
                   "bedeckt", "regnerisch"],
    "Temperatur": ["heiß", "heiß", "heiß", "warm", "kalt", "kalt", "kalt", "warm",
                   "kalt", "warm", "warm", "warm", "heiß", "warm"],
    "Luftfeuchtigkeit": ["hoch", "hoch", "hoch", "hoch", "normal", "normal", "normal",
                         "hoch", "normal", "normal", "normal", "hoch", "normal", "hoch"],
    "Wind": ["schwach", "stark", "schwach", "schwach", "schwach", "stark", "stark",
             "schwach", "schwach", "schwach", "stark", "stark", "schwach", "stark"],
    "Tennis?": ["nein", "nein", "ja", "ja", "ja", "nein", "ja", "nein",
                "ja", "ja", "ja", "ja", "ja", "nein"]
})

# Hypothesen für Tennis?
hypotheses = pd.DataFrame({
    "Vorhersage":  ["sonnig","regnerisch"],
    "Temperatur":  ["warm","kalt"],
    "Luftfeutigkeit": ["normal","hoch"],
    "Wind":  ["schwach","stark"],
    "Tennis?": ["ja","nein"]
})

# für funktionslernen

df = pd.DataFrame({
    "x": [1,1.2,2,2.5,3.1,3.7,4,0.5,0,-0.4,-1.1,-1.7,-2.3,-2.6,-3.3,-3.9],
    "y":[0.9,1.4,2.1,2.5,3,3.6,4.2,0.7,0,-0.6,-1,-1.4,-2,-2.4,-3.1,-4]
})

#data = np.array([[1,0.9],[1.2,1.4],[2,2.1],[2.5,2.5],[3.1,3],[3.7,3.6],[4,4.2],[0.5,0.7],
#                [0,0],[-0.4,-0.6],[-1.1,-1],[-1.7,-1.4],[-2.3,-2],[-2.6,-2.4],[-3.3,-3.1],
#                [-3.9,-4]])


print(funktionslernern(df,["1*x","x*x"]))
