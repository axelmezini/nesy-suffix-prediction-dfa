import re


class Model:
    """
    Loads a DECLARE model (.decl file), parses its constraints, and converts
    them into a single combined LTL formula as conjunction of all constraints
    """
    def __init__(self, folder_path, filename):
        self.folder_path = folder_path
        self.name = filename
        self.content = self.read_from_file()
        self.formulas = self.to_ltl()
        self.dfa = None

    def to_ltl(self):
        """
        Parses each non-comment line of the .decl file (format:
        "Template[activator]" or "Template[activator, target]"), converts
        each matched constraint to LTL, and joins them all with logical AND
        """
        formulas = []
        for row in self.content.split('\n'):
            if row and not row.startswith('#'):
                match = re.match(r'([\w\s-]+)\[([\w\s-]+)(?:, ([\w\s-]+))?]', row)
                if match:
                    declare_constraint = DeclareConstraint(match.group(1), match.group(2), match.group(3))
                    ltl_constraint = declare_constraint.to_ltl()
                    if ltl_constraint:
                        formulas.append(ltl_constraint)
        self.formulas = formulas
        return ' & '.join(self.formulas)

    def read_from_file(self):
        with open(str(self.folder_path / f'{self.name}.decl')) as f:
            return f.read()

    def write_formula_to_file(self):
        with open(str(self.folder_path / f'{self.name}_ltl.txt', 'w')) as f:
            f.write(' &\n'.join(self.formulas))


class DeclareConstraint:
    """
    Represents a single DECLARE constraint, a template plus one or two
    activity names, and knows how to translate itself into an LTL formula
    """
    def __init__(self, template, activator, target):
        self.name = template
        self.activation = clean_activity_name(activator)
        if target:
            self.target = clean_activity_name(target)

    def to_ltl(self):
        """
        Lookup table mapping each DECLARE template to its standard LTL
        translation; returns None for any unrecognized template name
        """
        if self.name == 'Init':
            return f'({self.activation})'
        elif self.name == 'Existence':
            return f'(F({self.activation}))'
        elif self.name == 'Existence2':
            return f'(F({self.activation} & X(F({self.activation}))))'
        elif self.name == 'Existence3':
            return f'(F({self.activation} & X(F({self.activation} & X(F({self.activation}))))))'
        elif self.name == 'Absence':
            return f'(!(F({self.activation})))'
        elif self.name == 'Absence2':
            return f'(!(F({self.activation} & X(F({self.activation}))    )))'
        elif self.name == 'Absence3':
            return f'(!(F({self.activation} & X(F({self.activation} & X(F({self.activation})))))))'
        elif self.name == 'Exactly1':
            return f'(F({self.activation}) & !(F({self.activation} & X(F({self.activation})))))'
        elif self.name == 'Exactly2':
            return f'(F({self.activation} & X(F({self.activation}))) & !(F({self.activation} & X(F({self.activation} & X(F({self.activation})))))))'
        elif self.name == 'Choice':
            return f'(F({self.activation}) | F({self.target}))'
        elif self.name == "Exclusive Choice":
            return f'((F({self.activation}) & !(F({self.target}))) | (F({self.target}) & !(F({self.activation}))))'
        elif self.name == 'Responded Existence':
            return f'(F({self.activation}) -> F({self.target}))'
        elif self.name == 'Co-Existence':
            return f'((F({self.activation}) -> F({self.target})) & (F({self.target}) -> F({self.activation})))'
        elif self.name == 'Response':
            return f'(G({self.activation} -> F({self.target})))'
        elif self.name == 'Alternate Response':
            return f'(G({self.activation} -> X(!({self.activation}) U {self.target})))'
        elif self.name == 'Chain Response':
            return f'(G({self.activation} -> X({self.target})))'
        elif self.name == 'Precedence':
            return f'((!({self.target}) U {self.activation}) | G(!({self.target})))'
        elif self.name == 'Alternate Precedence':
            return f'((((!{self.target} U {self.activation}) | G(!{self.target})) & G({self.target} ->((!(X({self.activation})) & !(X(!({self.activation})))) | X((!({self.target}) U {self.activation}) | G(!({self.target})))))) & !({self.target}))'
        elif self.name == 'Chain Precedence':
            return f'(G(X({self.target}) -> {self.activation}) & !({self.target}))'
        elif self.name == 'Succession':
            return f'(G({self.activation} -> F({self.target})) & ((!({self.target}) U {self.activation}) | G(!({self.target}))))'
        elif self.name == 'Alternate Succession':
            return f'(G({self.activation} -> X((!({self.activation})) U {self.target})) & (((!({self.target}) U {self.activation}) | G(!({self.target}))) & G({self.target} -> ((!(X({self.activation})) & !(X(!({self.activation})))) | X((!({self.target}) U {self.activation}) | G(!({self.target}))))) & !({self.target})))'
        elif self.name == 'Chain Succession':
            return f'(G({self.activation} -> X({self.target})) & G(X({self.target}) -> {self.activation}) & !({self.target}))'
        elif self.name == 'Not Responded Existence':
            return f'(F({self.activation}) -> !(F({self.target})))'
        elif self.name == 'Not Co-Existence':
            return f'(!(F({self.activation}) & F({self.target})))'
        elif self.name in ['Not Response', 'Not Precedence', 'Not Succession']:
            return f'G({self.activation} -> !(F({self.target})))'
        elif self.name in ['Not Chain Response', 'Not Chain Precedence', 'Not Chain Succession']:
            return f'(G({self.activation} -> !(X({self.target}))))'
        else:
            return None

def clean_activity_name(name):
    """
    Normalizes an activity name into a valid LTL atom identifier
    (lowercase, spaces/dashes/dots/parens replaced with underscores, "a_" prefixed)
    """
    return f"a_{name.lower().replace(' ', '_').replace('-', '_').replace('.', '_').replace('(', '_').replace(')', '_')}"
