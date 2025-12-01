import numpy as np
import pandas as pd
import pytest

from howso.client.schemas.reaction import Reaction


def test_cases_with_details_add_reaction():
    """Tests that `Reaction` `add_reaction` works with different data types."""
    df = pd.DataFrame(data=np.asarray([
        ['a', 'b', 'c', 'd'],
        ['2020-9-12T9:09:09.123', '2020-10-12T10:10:10.333',
            '2020-12-12T12:12:12.444', '2020-10-11T11:11:11.222']
    ]).transpose(), columns=['nom', 'datetime'])

    react_response = {
        'details': {'action_features': ['datetime']},
        'action': df
    }

    cwd = Reaction(react_response['action'], react_response['details'])
    cwd.add_reaction(react_response['action'].to_dict(), react_response['details'])
    # List of dicts
    cwd.add_reaction(react_response['action'].to_dict(orient='records'), react_response['details'])
    cwd.add_reaction(Reaction(react_response['action'], react_response['details']))

    assert cwd['action'].shape[0] == 16


def test_cases_with_details_instantiate():
    """Tests that `Reaction` can be instantiated with different data types."""
    df = pd.DataFrame(data=np.asarray([
        ['a', 'b', 'c', 'd'],
        ['2020-9-12T9:09:09.123', '2020-10-12T10:10:10.333',
            '2020-12-12T12:12:12.444', '2020-10-11T11:11:11.222']
    ]).transpose(), columns=['nom', 'datetime'])

    react_response = {
        'details': {'action_features': ['datetime']},
        'action': df
    }

    cwd = Reaction(react_response['action'], react_response['details'])
    assert cwd['action'].shape[0] == 4

    cwd = Reaction(react_response['action'].to_dict(), react_response['details'])
    assert cwd['action'].shape[0] == 4

    cwd = Reaction(react_response['action'].to_dict(orient='records'), react_response['details'])
    assert cwd['action'].shape[0] == 4
