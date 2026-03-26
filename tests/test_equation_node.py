from nodes.node_equations import extract_equations


def test_equation_validation_filters_noise():
    text = "Price is $abc$ but equation is $E=mc^2$ and \\[a+b=c\\]"
    eqs = extract_equations(text)
    assert "E=mc^2" in eqs
    assert "abc" not in eqs
