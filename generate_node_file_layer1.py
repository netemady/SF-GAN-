# netemady - 042721
# generate .node file from infomap output for visualization with BnetViewer

#read infomap output and find total num. of modules

invalid_subjects = []

for subject in range(25):

        #name = 'input_day1_sub{}_upperT.txt_states.clu'.format(subject)
        name = 'D:/Negar/GT_GAN/Infomap-Strategies/day1/Str2/output/input_day1_sub{}_strategy2.txt_states.clu'.format(subject)

        f = open(name, 'r')
        temp = f.readlines()
        all_modules = []
        for l in temp:
            l_str = l.strip()
            l_str_sp = l_str.split(' ')

            if l_str_sp[0] != '#' and l_str != '':
                all_modules.append(int(l_str_sp[1]))
        f.close()

        num_of_modules = max(all_modules)

        if num_of_modules > 21:
            invalid_subjects.append(subject)
            continue
        else:

            # pull coordinates from a sample node file
            node_file_lines = []
            z_values = []

            f = open('new_Desikan_Killiany_sample_to_visualize.txt', 'r')
            lines = f.readlines()
            for line in lines:
                node_file_lines.append(line.strip())

                line_elements = line.strip().split(' ')
                if line_elements[2] != '68':
                    z_values.append(float(line_elements[2]))

            f.close()

            # print(max(z_values)) #57.207729
            # print(min(z_values)) #-34.628571
            # print(max(z_values)-min(z_values)) #91.8363

            layer1_modules = []

            # find nodes belonging to each module in layer 1 and generate a temporary node file
            f = open('Desikan_Killiany_temporary_layer1.txt', 'w')

            for l in temp:
                l_str = l.strip()
                l_str_sp = l_str.split(' ')
                if l_str_sp[-1] == '1' and l_str_sp[0] != '#':  # layer1
                    node_id = l_str_sp[-2]
                    module = l_str_sp[1]
                    layer1_modules.append(int(module))
                    line_ = node_file_lines[int(node_id)]
                    line_el = line_.split(' ')
                    line_new = line_el[0] + ' ' + line_el[1] + ' ' + line_el[2] + ' ' + str(
                        module) + '.000000' + ' ' + '2.000000' + ' ' + line_el[5]
                    f.write(line_new + '\n')

            f.close()

            # find missing modules

            missing_modules = []

            for nofm in range(max(layer1_modules)):

                if nofm + 1 not in layer1_modules:
                    missing_modules.append(nofm + 1)

            # organize the node file according to left/right and order of regions

            region_names = ['bankssts', 'caudalanteriorcingulate', 'caudalmiddlefrontal', 'cuneus', 'entorhinal', 'fusiform', 'inferiorparietal', \
                            'inferiortemporal', 'isthmuscingulate', 'lateraloccipital', 'lateralorbitofrontal', \
                            'lingual', 'medialorbitofrontal', 'middletemporal', 'parahippocampal', 'paracentral', 'parsopercularis', 'parsorbitalis', \
                            'parstriangularis', 'pericalcarine', 'postcentral', 'posteriorcingulate', 'precentral', 'precuneus', 'rostralanteriorcingulate', \
                            'rostralmiddlefrontal', 'superiorfrontal', 'superiorparietal', 'superiortemporal', 'supramarginal', 'frontalpole', 'temporalpole', 'transversetemporal', 'insula']

            f = open('Desikan_Killiany_temporary_layer1.txt', 'r')
            DK_temp_lines = f.readlines()
            f.close()

            value1 = -34.6284

            # new_name = 'D:/Negar/GT_GAN/Infomap-Strategies/Str3/matlab-s3/layer1/day1_sub{}_Strategy3'.format(subject)
            for z_figure in range(23):
                # values = [-34.6284, -16.2584, 2.1116, 20.4816, 38.8517, 57.2217]

                min_value = value1
                max_value = value1 + (z_figure + 1) * 4
                value1 = max_value

                new_name = 'D:/Negar/GT_GAN/Infomap-Strategies/day1/Str2/matlab-s2/layer1/multiple-z/day1_sub{}_h{}_Strategy2'.format(subject, z_figure)

                f = open(new_name + '_to_Bnet_L1.node', 'w')

                for s in region_names:
                    for line in DK_temp_lines:
                        line2 = line.strip()
                        line3 = line2.split(' ')
                        if line3[-1] == 'l.' + s:
                            if min_value <= float(line3[2]) <= max_value:
                                f.write(line2 + '\n')
                                break
                            else:
                                f.write(line3[0] + ' ' + line3[1] + ' ' + line3[2] + ' ' + line3[3] + ' ' + '0.000500' + ' ' + line3[-1] + '\n')
                                break

                for s in region_names:
                    for line in DK_temp_lines:
                        line2 = line.strip()
                        line3 = line2.split(' ')
                        if line3[-1] == 'r.' + s:
                            if min_value <= float(line3[2]) <= max_value:
                                f.write(line2 + '\n')
                                break
                            else:
                                f.write(line3[0] + ' ' + line3[1] + ' ' + line3[2] + ' ' + line3[3] + ' ' + '0.000500' + ' ' + line3[-1] + '\n')
                                break

                for n in range(len(missing_modules)):
                    f.write('0.000000' + ' ' + '0.000000' + ' ' + '0.000000' + ' ' + str(missing_modules[n]) + '.000000' + ' ' + '0.000500' + '\n')

                f.close()

print(invalid_subjects)



