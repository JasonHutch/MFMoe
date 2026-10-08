def build_prompt_for_theme(base:str, theme:str, pos_examples:list[str], neg_examples:list[str]):
    pos_string = ",".join(pos_examples)
    neg_string = ",".join(neg_examples)

    b_theme = base.replace("<THEME>", theme)
    b_pos = b_theme.replace("<POS EXAMPLES>", pos_string)
    final = b_pos.replace("<NEG EXAMPLES>", neg_string)

    return final


