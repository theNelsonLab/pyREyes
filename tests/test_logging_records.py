"""`format_record`: the `KEY | field=val | ...` shape of REyes machine records."""

from pyREyes.lib.REyes_logging import format_record


def test_format_record_never_breaks_the_field_split():
    line = format_record('X', ('a', None), ('b', 'p | q\nr'), ('c', [1, 2]),
                         ('d', []), ('e', True))
    assert line == 'X | a=NA | b=p / q r | c=1,2 | d=NA | e=true'


def test_a_record_with_no_fields_is_just_its_key():
    assert format_record('DONE') == 'DONE'


def test_every_field_is_present_and_splits_the_same_way():
    line = format_record('STAGE', ('item', 7), ('reason', 'valve closed'),
                         ('log', None), ('ok', False), ('pair', (1.5, 'x')))
    fields = line.split(' | ')
    assert fields[0] == 'STAGE'
    assert dict(f.split('=', 1) for f in fields[1:]) == {
        'item': '7', 'reason': 'valve closed', 'log': 'NA', 'ok': 'false',
        'pair': '1.5,x'}
