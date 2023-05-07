from pyhealth.data import Patient, Visit

def drug_recommendation_mimic4_no_hist(patient: Patient):
    samples = []
    for i in range(len(patient)):
        visit: Visit = patient[i]
        conditions = visit.get_code_list(table="diagnoses_icd")
        procedures = visit.get_code_list(table="procedures_icd")
        drugs = visit.get_code_list(table="prescriptions")
        # ATC 3 level
        drugs = [drug[:4] for drug in drugs]
        # exclude: visits without condition, procedure, or drug code
        if len(conditions) * len(procedures) * len(drugs) == 0:
            continue
        # TODO: should also exclude visit with age < 18
        samples.append(
            {
                "visit_id": visit.visit_id,
                "patient_id": patient.patient_id,
                "conditions": conditions,
                "procedures": procedures,
                "drugs": drugs,
                "drugs_all": drugs,
            }
        )
    # exclude: patients with less than 2 visit
    if len(samples) < 2:
        return []
    # dont add history, just make lists
    samples[0]["conditions"] = [samples[0]["conditions"]]
    samples[0]["procedures"] = [samples[0]["procedures"]]
    samples[0]["drugs_all"] = [samples[0]["drugs_all"]]

    for i in range(1, len(samples)):
        samples[i]["conditions"] = [samples[i]["conditions"]]
        samples[i]["procedures"] = [samples[i]["procedures"]]
        samples[i]["drugs_all"] = [samples[i]["drugs_all"]]

    return samples

def drug_recommendation_mimic4_no_proc(patient: Patient):
    samples = []
    for i in range(len(patient)):
        visit: Visit = patient[i]
        conditions = visit.get_code_list(table="diagnoses_icd")
        #procedures = visit.get_code_list(table="procedures_icd")
        drugs = visit.get_code_list(table="prescriptions")
        # ATC 3 level
        drugs = [drug[:4] for drug in drugs]
        # exclude: visits without condition, procedure, or drug code
        #if len(conditions) * len(procedures) * len(drugs) == 0:
        if len(conditions) * len(drugs) == 0:
            continue
        # TODO: should also exclude visit with age < 18
        samples.append(
            {
                "visit_id": visit.visit_id,
                "patient_id": patient.patient_id,
                "conditions": conditions,
                #"procedures": procedures,
                "drugs": drugs,
                "drugs_all": drugs,
            }
        )
    # exclude: patients with less than 2 visit
    if len(samples) < 2:
        return []
    # add history
    samples[0]["conditions"] = [samples[0]["conditions"]]
    #samples[0]["procedures"] = [[1]]
    samples[0]["drugs_all"] = [samples[0]["drugs_all"]]

    for i in range(1, len(samples)):
        samples[i]["conditions"] = samples[i - 1]["conditions"] + [
            samples[i]["conditions"]
        ]
        #samples[i]["procedures"] = samples[i - 1]["procedures"] + [[1]]
        samples[i]["drugs_all"] = samples[i - 1]["drugs_all"] + [
            samples[i]["drugs_all"]
        ]

    return samples

## TODO: THIS DOESNT SEEM TO WORK FOR RNN
## IT ALSO TAKES A VERY LONG TIME
## I ALSO THINK THIS IS NOT GOING TO WORK THE WAY I EXPECT, SO THE LOGIC NEEDS TO BE DIFFERENT
#def drug_recommendation_mimic4_flat(patient: Patient):
#    samples = []
#    for i in range(len(patient)):
#        visit: Visit = patient[i]
#        conditions = visit.get_code_list(table="diagnoses_icd")
#        procedures = visit.get_code_list(table="procedures_icd")
#        drugs = visit.get_code_list(table="prescriptions")
#        # ATC 3 level
#        drugs = [drug[:4] for drug in drugs]
#        # exclude: visits without condition, procedure, or drug code
#        if len(conditions) * len(procedures) * len(drugs) == 0:
#            continue
#        # TODO: should also exclude visit with age < 18
#        samples.append(
#            {
#                "visit_id": visit.visit_id,
#                "patient_id": patient.patient_id,
#                "conditions": conditions,
#                "procedures": procedures,
#                "drugs": drugs,
#                "drugs_all": drugs,
#            }
#        )
#    # exclude: patients with less than 2 visit
#    if len(samples) < 2:
#        return []
#    # add history
#    # also, convert procedures and conditions to sets
#    # then back to list so we just have one instance of conditions procedures
#    samples[0]["conditions"] = list(set(samples[0]["conditions"]))
#    samples[0]["procedures"] = list(set(samples[0]["procedures"]))
#    #samples[0]["drugs_all"] = [samples[0]["drugs_all"]]
#    samples[0]["drugs_all"] = list(set(samples[0]["drugs_all"]))
#
#    for i in range(1, len(samples)):
#        samples[i]["conditions"] = list(set(samples[i - 1]["conditions"] +
#            samples[i]["conditions"]
#        ))
#        samples[i]["procedures"] = list(set(samples[i - 1]["procedures"] +
#            samples[i]["procedures"]
#        ))
#        #samples[i]["drugs_all"] = samples[i - 1]["drugs_all"] + [
#        #    samples[i]["drugs_all"]
#        #]
#        samples[i]["drugs_all"] = list(set(samples[i - 1]["drugs_all"] +
#            samples[i]["drugs_all"]
#        ))
#
#    #print(samples[0])
#    #print(samples[1])
#
#    for i in range(0, len(samples)):
#        samples[i]["conditions"] = [samples[i]["conditions"]]
#        samples[i]["procedures"] = [samples[i]["procedures"]]
#        samples[i]["drugs_all"] = [samples[i]["drugs_all"]]
#
#    #print(samples[0])
#    #print(samples[1])
#
#    return samples
