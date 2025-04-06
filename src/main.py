import utils as ut

def main():

    file_path = ut.parse_args()[0]

    model = ut.load_model(file_path)

    species_list = ut.get_list_of_species(model)

    reactions_list = ut.get_list_of_reactions(model)

    r_dict = ut.get_reactants_dict(reactions_list, ut.species_dict(species_list))
    p_dict = ut.get_products_dict(reactions_list, ut.species_dict(species_list))

    ut.dict_pretty_print(p_dict)

if __name__ == '__main__':
    main()
