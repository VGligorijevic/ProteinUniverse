import numpy as np
from Bio import Align
from Bio.Data.SCOPData import protein_letters_3to1
from Bio.SeqUtils import seq1

TEN_ANGSTROMS = 10.0
ALIGNED_BY_SEQRES = 'aligned by SEQRES'
ATOMS_ONLY = 'ATOM lines only'


class ContactMapContainer:
    def __init__(self):
        self.chains = {}

    def with_chain(self, chain_name):
        self.chains[chain_name] = {}

    def with_chain_seq(self, chain_name, seq):
        self.chains[chain_name]['seq'] = seq

    def with_map_for_chain(self, chain_name, contact_map):
        self.chains[chain_name]['contact-map'] = contact_map

    def with_alignment_for_chain(self, chain_name, alignment):
        self.chains[chain_name]['alignment'] = alignment

    def with_method_for_chain(self, chain_name, method):
        self.chains[chain_name]['method'] = method

    def with_final_seq_for_chain(self, chain_name, final_seq):
        self.chains[chain_name]['final-seq'] = final_seq


def correct_residue(x, target):
    try:
        sl = protein_letters_3to1[x.resname]
        if sl == target:
            return True
        return False
    except KeyError:
        return False


class ContactMapBuilder:
    def __init__(self):
        pass

    def generate_contact_map_for_pdb(self, structure_container, pedantic_mode=True, verbose=True):
        aligner = Align.PairwiseAligner()
        contact_maps = ContactMapContainer()

        model = structure_container.structure[0]

        for chain_name in structure_container.chains:
            chain = structure_container.chains[chain_name]
            contact_maps.with_chain(chain_name)
            if verbose:
                print(f"\nProcessing chain {chain_name}")

            if chain['seqres-seq'] is not None and len(chain['seqres-seq']) > 0:
                contact_maps.with_method_for_chain(chain_name, ALIGNED_BY_SEQRES)
                seqres_seq = chain['seqres-seq']
                atom_seq = chain['atom-seq']

                alignment = aligner.align(seqres_seq, atom_seq)
                specific_alignment = next(alignment)
                if verbose:
                    print(f"Seqres seq: {seqres_seq}")
                    print(f"Atom seq:   {atom_seq}")
                    print(specific_alignment)

                contact_maps.with_alignment_for_chain(chain_name, specific_alignment)

                # It's actually much easier to just have biopython generate the string alignment
                # and use that as a guide.
                pattern = specific_alignment.__str__().split("\n")
                aligned_seqres_seq = pattern[0]
                mask = pattern[1]
                aligned_atom_seq = pattern[2]

                # Build a list of residues that we do have atoms for.
                reindexed_residues = []
                residues = model[chain_name].get_residues()
                for r in residues:
                    reindexed_residues.append(r)

                final_residue_list = []
                picked_residues = 0
                non_canonicals_or_het = 0

                for i in range(len(aligned_atom_seq)):
                    if aligned_seqres_seq[i] == '-':
                        # This is an inserted residue from the aligner that doesn't actually match any
                        # seqres line. Don't even insert a None.
                        continue
                    current_aligned_atom_residue_letter = aligned_atom_seq[i]
                    #  atom seq has a letter and the mask shows it corresponds to a seqres item
                    if current_aligned_atom_residue_letter != '-' and mask[i] == '|':
                        candidate_residue = next((x for x in reindexed_residues[picked_residues:picked_residues + 5] if
                                                  correct_residue(x, current_aligned_atom_residue_letter)), None)

                        if candidate_residue is None:
                            # The right answer is probably 'None' but we need to know why.
                            residue = reindexed_residues[picked_residues]
                            if residue.id[0].startswith('H_'):
                                non_canonicals_or_het += 1
                        else:
                            picked_residues += 1

                        final_residue_list.append(candidate_residue)
                    else:
                        final_residue_list.append(None)

                final_seq_three_letter_codes = ''.join(
                    [r.resname if r is not None else 'XXX' for r in final_residue_list])
                final_seq_one_letter_codes = seq1(final_seq_three_letter_codes, undef_code='-',
                                                  custom_map=protein_letters_3to1)
                if verbose:
                    print(f"Final [len of seq {len(seqres_seq)}] [len of result {len(final_seq_one_letter_codes)}] "
                          f"[len of final residue list {len(final_residue_list)}]:\n{final_seq_one_letter_codes}")

                if len(final_residue_list) != len(seqres_seq) and pedantic_mode:
                    raise Exception(
                        f"Somehow the final residue list {len(final_residue_list)} doesn't match the size of the SEQRES seq {len(seqres_seq)}")

                if (len(seqres_seq) != len(final_seq_one_letter_codes) != len(final_residue_list)) and pedantic_mode:
                    raise Exception(
                        'The length of the SEQRES seq != length of final_seq_one_letter_codes != length of final residue list')

                sanity_check = aligned_atom_seq.replace('X', '')
                if sanity_check != final_seq_one_letter_codes and pedantic_mode:
                    print(f"sanity_check {sanity_check}")
                    print(f"final_seq    {final_seq_one_letter_codes}")
                    count = sum(1 for a, b in zip(sanity_check, final_seq_one_letter_codes) if a != b)
                    # While going through the data we found some _very_ large structures in the PDB.
                    # Some of them have massive interior chains w/ tons of missing data. In this case
                    # we're basically just saying we did what we could, passing the data along and saying
                    # we still were in pedantic mode.
                    missing_residue_heuristic = sanity_check.count('-') / len(sanity_check)
                    missing_residue_heuristic_2 = final_seq_one_letter_codes.count('-') / len(final_seq_one_letter_codes)
                    if count == non_canonicals_or_het:
                        # Add a message about this.
                        print(
                            f"Warning: The final sequence and the sanity check were different, but the difference equals the number of HETATMs or non-canonical residues. _Probably_ OK.")
                    elif missing_residue_heuristic >= 0.5 or missing_residue_heuristic_2 >= 0.5:
                        print(f"Warning: The final sequence and the sanity check were different. Over 50% of the chain is unresolved. Nothing we can do about it.")
                    else:
                        raise Exception(
                            f'The final one letter SEQ generated from residues does not match the aligned atom seq (Diff count {count} but HETATM {non_canonicals_or_het})')

                contact_maps.with_final_seq_for_chain(chain_name, final_seq_one_letter_codes)
                contact_maps.with_chain_seq(chain_name, seqres_seq)
                contact_maps.with_map_for_chain(chain_name,
                                                self.__residue_list_to_contact_map(final_residue_list, len(seqres_seq)))
            else:
                contact_maps.with_method_for_chain(chain_name, ATOMS_ONLY)
                atom_seq = chain['atom-seq']
                residues = model[chain_name].get_residues()

                final_residue_list = []
                missing_alpha_carbons = []
                for r in residues:
                    try:
                        _ = r["CA"]
                        final_residue_list.append(r)
                    except KeyError:
                        missing_alpha_carbons.append(r)

                # Sanity checks
                final_seq_three_letter_codes = ''.join(
                    [r.resname if r is not None else 'XXX' for r in final_residue_list])
                final_seq_one_letter_codes = seq1(final_seq_three_letter_codes, undef_code='-',
                                                  custom_map=protein_letters_3to1)
                print(final_seq_one_letter_codes)
                corrected_atom_seq = final_seq_one_letter_codes
                # End sanity checks

                contact_maps.with_chain_seq(chain_name, corrected_atom_seq)
                contact_maps.with_map_for_chain(chain_name,
                                                self.__residue_list_to_contact_map(final_residue_list, len(corrected_atom_seq)))

        return contact_maps

    def __residue_list_to_contact_map(self, residue_list, length):
        dist_matrix = self.__calc_dist_matrix(residue_list)
        diag = self.__diagnolize_to_fill_gaps(dist_matrix, length)
        contact_map = self.__create_adj(diag, TEN_ANGSTROMS)
        return contact_map

    def __norm_adj(self, A):
        #  Normalize adj matrix.
        with np.errstate(divide='ignore'):
            d = 1.0 / np.sqrt(A.sum(axis=1))
        d[np.isinf(d)] = 0.0

        # normalize adjacency matrices
        d = np.diag(d)
        A = d.dot(A.dot(d))

        return A

    def __create_adj(self, _A, thresh):
        # Create CMAP from distance
        A = _A.copy()
        with np.errstate(invalid='ignore'):
            A[A <= thresh] = 1.0
            A[A > thresh] = 0.0
            A[np.isnan(A)] = 0.0
            A = self.__norm_adj(A)

        return A

    def __calc_residue_dist(self, residue_one, residue_two):
        """Returns the C-alpha distance between two residues"""
        if residue_one is None:
            return 10000.0
        if residue_two is None:
            return 10000.0
        try:
            diff_vector = residue_one["CA"].coord - residue_two["CA"].coord
            return np.sqrt(np.sum(diff_vector * diff_vector))
        except KeyError:
            return 1000.0

    def __diagnolize_to_fill_gaps(self, distance_matrix, length):
        # Create CMAP from distance
        A = distance_matrix.copy()
        for i in range(length):
            if A[i][i] == 10000.0:
                A[i][i] = 1.0
                try:
                    A[i + 1][i] = 1.0
                except IndexError:
                    pass
                try:
                    A[i][i + 1] = 1.0
                except IndexError:
                    pass

        return A

    def __calc_dist_matrix(self, chain_one):
        """Returns a matrix of C-alpha distances between two chains"""
        answer = np.zeros((len(chain_one), len(chain_one)), np.float)
        for row, residue_one in enumerate(chain_one):
            for col, residue_two in enumerate(chain_one[row:], start=row):
                if col >= len(chain_one):
                    continue  # enumerate syntax is convenient, but results in invalid indices on last column
                answer[row, col] = self.__calc_residue_dist(residue_one, residue_two)
                answer[col, row] = answer[row, col]  # cchandler
        return answer
