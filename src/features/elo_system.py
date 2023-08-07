import numpy as np


class EloSystem():
    """
    Providing a custom ELO interface
    """
    def __init__(self, base_rating=1500) -> None:
        self.base_rating = base_rating
        self.fighters = {}

    def add_fighter(self, name):
        self.fighters[name] = self.base_rating

    def add_match(self, fighter1, fighter2, winner, k=16):
        
        if fighter1 not in self.fighters.keys():
            self.add_fighter(fighter1)
        if fighter2 not in self.fighters.keys():
            self.add_fighter(fighter2)

        Ra, Rb = self.fighters[fighter1], self.fighters[fighter2]

        Ea = 1 / (1 + 10**((Rb-Ra) / 400))
        Eb = 1 / (1 + 10**((Ra-Rb) / 400))

        if winner == 'W':
            Sa = 1
            Sb = 0
        elif winner == 'L':
            Sa = 0
            Sb = 1
        elif winner == 'D':
            Sa = 0.5
            Sb = 0.5
        else:
            raise Exception('Winner not provided: A for Fighter 1, B for Fighter 2, or T for tie')

        Ra_adj = Ra + k * (Sa - Ea)
        Rb_adj = Rb + k * (Sb - Eb)

        self.fighters[fighter1], self.fighters[fighter2] = Ra_adj, Rb_adj
