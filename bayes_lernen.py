import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go


def bayes_theorem(probabilities):
    """
    P(h|D) = (P(D|h)*P(h)) / P(D)
    P(h) - Wahrscheinlichkeit das h gültig ist
    P(D) - Wahrscheinlichkeit, dass D als Ereignisdatensatz auftritt
    h - Hypothse
    D - Datensatz
    :return: P(h|D) (a posteriori Wahrscheinlichkeit)
    """
    # P(D) = P(D|h) * P(h) + P(D|!h) * P(!h)
    p_d = probabilities["P(D|h)"] * probabilities["P(h)"] + probabilities["P(D|!h)"] * probabilities["P(!h)"]

    return (probabilities["P(D|h)"] * probabilities["P(h)"]) / p_d


def maximum_a_posteriori_hypothese(hypotheses, data):
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


def konzeptlernen(train_data, hypotheses):
    """
    ges. c: x->{0,1}
    c - Zielhypothese
    Abbildung des Datensatz(x) auf False(0) order True(1) (Bsp. Wir wollen etwas machen oder nicht)
    :param hypotheses:
    :param train_data: x_i mit gegebenem c(x_i)
    :return:
    """
    p_h_gegeben_D = []
    for hypo in hypotheses.index:
        for idx in train_data.index:
            if train_data.at[idx, "Tennis?"] == hypotheses.at[hypo, "Tennis?"]:
                train_data.at[idx, "P_D_gegeben_h"] = 1
            else:
                train_data.at[idx, "P_D_gegeben_h"] = 0

        p_h_gegeben_D.append(1 / sum(df["P_D_gegeben_h"]))

    return p_h_gegeben_D


def print_data(data):
    t = np.linspace(-4, 4, 200)

    fig = px.scatter(data, "x", "y")
    fig.add_trace(
        go.Scatter(
            x=[-4, 4],
            y=[-4, 4],
            mode="lines",
            showlegend=False
        )
    )
    fig.add_trace(
        go.Scatter(
            x=t,
            y=t * t,
            showlegend=False
        )
    )
    fig.show()


def functions(function, value):
    if function == "1*x":
        return value
    elif function == "x*x":
        return value * value


def funktionslernern(train_data, funktions_hypothesen):
    """
    gesucht wird, eine reell wertige Zielfunktion
    gegeben sind, die Punkte x_i und d_i
    d_i = f(x_i) + e_i
    e_i - normalverteilte zufallsvaribale

    h_ml = argmin( sum(d_i - h(x_i))^2)
    :param funktions_hypothesen:
    :param train_data:
    :return:
    """

    # Messfehler hinzufügen
    train_data["error"] = np.random.randn(len(train_data))
    train_data["d_i"] = train_data["y"] + train_data["error"]
    # target values bestimmen
    for hypo in funktions_hypothesen:
        for idx in train_data.index:
            erg = functions(hypo, train_data.at[idx, "x"])
            train_data.at[idx, hypo] = erg

    fehlerquadrate = []

    for hypothese in funktions_hypothesen:
        total = 0
        for idx in train_data.index:
            erg = (train_data.at[idx, "d_i"] - train_data.at[idx, hypothese]) ** 2
            total += erg

        fehlerquadrate.append(erg)

    h_ML_idx = np.argmin(fehlerquadrate)
    h_ML = funktions_hypothesen[h_ML_idx]

    print_data(train_data)

    return h_ML


def p_kj_gegeben_hi(k_j, h_i):
    if h_i == k_j:
        return 1
    else:
        return 0


def optimaler_bayes(data):
    """
    h_map ist nicht zwingend die wahrscheinlichste Klassifikation
    Nun gesucht die wahrscheinlichste Klassifikation v_j einer neuen Instanz x
    k_OB = argmax k (sum(P(k_j|h_i)*P(h_i|D)))
    :param data:
    :return:
    """
    # mögliche Ausgänge
    k_j = [True, False]

    probabilities = []

    for k in k_j:
        classificator = 0
        for i in range(len(data["h_i(x)"])):
            erg = p_kj_gegeben_hi(k, data["h_i(x)"][i]) * data["P(h_i|D)"][i]
            classificator += erg
        probabilities.append(classificator)

    k_ob_idx = np.argmax(probabilities)
    k_ob = k_j[k_ob_idx]

    return k_ob


def count(data, tuple1, tuple2):
    # tuple = (col, ausprägung)

    counts = 0
    for idx in data.index:
        if data.at[idx, tuple1[0]] == tuple1[1] and data.at[idx, tuple2[0]] == tuple2[1]:
            counts += 1

    erg = counts / data[tuple2[0]].to_list().count(tuple2[1])

    return erg


def naiver_bayes_classificator(data, new_instance):
    """
    gegeben:
    Instanz x: Konjunktion von Attributen <a_1,...,a_n>
    Endlichr Menge von Klassen V = {v1,...,v_m}
    Menge klassifizierter Beispiele
    gesucht:
    v_map = argmax(P(a_1,...,a_n|v_j)*P(v_j))
    P(a_1,...,a_n|v_j) = P(a_1|v_j)*...*P(a_n|v_j)
    nehmen der Wahrscheinlichsten Klasse

    wenn P(a_i|v_j) nicht gegeben dann schätzen durch
    P(a_i|v_j) = (n_c + m*p) / (n+m)
    n - Anzahl Attribute
    m - Anzahl Klassen
    n_c - Anzahl der Objekte in Klasse v_j mit Attribut a_i
    p - 1/ Anzahl der Klassen

    :return:
    """

    anzahl_data = len(data)
    # Klassen V={Tennis?ja,Tennis?nein}
    classes = list(set(data[
                           "Tennis?"].values))  # Umwandlung in Liste damit wieder reihnfolge gegebne ist für später identifizierung der max stelle

    # Atrribute : attributausprägung
    attributes = {col: set(data[col].values) for col in data.columns}
    del attributes["Tennis?"]

    # Berechnen der Wahrscheinlichkteit für Auftreten einer Klasse
    p_vj = {}

    for cl in classes:
        p_vj[cl] = data["Tennis?"].to_list().count(cl) / anzahl_data

    # Berechnen/Schätzen der Wahrscheinlichkeiten P(a_i|v_j)
    p_ai_gegeben_vj = {}

    for cls in attributes:
        for atr in attributes[cls]:
            for cl in classes:
                event = atr + '|' + cl
                prob = count(data, (cls, atr), ("Tennis?", cl))
                p_ai_gegeben_vj[event] = prob

    p_classification = []
    for cl in list(classes):
        prob = p_vj[cl]
        for atr in new_instance:
            event = new_instance[atr] + '|' + cl
            prob *= p_ai_gegeben_vj[event]
        p_classification.append(prob)

    argmax_idx = np.argmax(p_classification)
    p_classification = classes[argmax_idx]

    # ggf. Wahrscheinlichkeit normieren

    return p_classification


# h-Krebs, D- test positiv
data = {"P(h)": 0.008, "P(!h)": 0.992, "P(D|h)": 0.98,
        "P(!D|h)": 0.02, "P(D|!h)": 0.03, "P(!D|!h)": 0.97}

# Beobachtung neuer Patient, Test positiv. Hat der neue Patient Krebs?
hypotheses = [["P(D|h)", "P(h)"], ["P(D|!h)", "P(!h)"]]

# möglich für Konzept lernen?
# verwendet für naiver bayscher Klassifcator
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

neue_instanz = {"Vorhersage": "sonnig", "Temperatur": "kalt", "Luftfeutigkeit": "hoch",
                "Wind": "stark"}

# Hypothesen für Tennis?
hypotheses = pd.DataFrame({
    "Vorhersage": ["sonnig", "regnerisch"],
    "Temperatur": ["warm", "kalt"],
    "Luftfeutigkeit": ["normal", "hoch"],
    "Wind": ["schwach", "stark"],
    "Tennis?": ["ja", "nein"]
})

# für funktionslernen

df = pd.DataFrame({
    "x": [1, 1.2, 2, 2.5, 3.1, 3.7, 4, 0.5, 0, -0.4, -1.1, -1.7, -2.3, -2.6, -3.3, -3.9],
    "y": [0.9, 1.4, 2.1, 2.5, 3, 3.6, 4.2, 0.7, 0, -0.6, -1, -1.4, -2, -2.4, -3.1, -4]
})

# für optimaler Bayes
data = {"P(h_i|D)": [0.4, 0.3, 0.3], "h_i(x)": [True, False, False]}
