# -*- coding: utf-8 -*-

import re
sentence = "I was born in the year 1998"
re.match(r".*",sentence)          # Here '.' means any character and '*' means zero or more
re.match(r".+",sentence)          # Here '+' means one or more

re.match(r"[a-zA-Z]*",sentence)   # match function returns the first match only


sentence1 = "a"
re.match(r"ab?",sentence1)        # There has to be exactly one 'b' after 'a' or no 'b'