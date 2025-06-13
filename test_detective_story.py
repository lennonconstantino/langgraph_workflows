from detective_story import detective_workflow

def test_detective_story():
    result = detective_workflow.invoke({})

    print("\n=== HISTÓRIA DO DETETIVE ===\n")
    print("\n=== PRIMEIRO ATO ===\n")
    print(result["story"]["act_1"])
    print(" \n=== SEGUNDO ATO ===\n")
    print(result["story"]["act_2"])
    print("\n=== TERCEIRO ATO ===\n")
    print(result["story"]["act_3"])
    print("\n=== QUARTO ATO ===\n")
    print(result["story"]["act_4"])

    print("\n=== ELEMENTOS DA HISTÓRIA ===\n")
    print(f"Detetive: {result['detective']}")
    print(f"Crime: {result['crime']}")
    print(f"Local: {result['location']}")
    print(f"pista Inicial: {result['clue']}")

if __name__ == "__main__":
    test_detective_story()