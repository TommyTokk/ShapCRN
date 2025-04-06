import utils as ut

def main():

    file_path = ut.parse_args()[0]

    model = ut.load_model(file_path)

    species_list = ut.get_list_of_species(model)

    species_dict_list = []

    for specie in species_list:
        species_dict_list.append(ut.specie_to_dict(specie))

    print (species_dict_list)

if __name__ == '__main__':
    main()
