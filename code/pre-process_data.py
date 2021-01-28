import json
import write_data
import data_tools
import argparse

###########################
#  get TRAINING DATA      #
###########################

def main():

    parser = argparse.ArgumentParser()

    parser.add_argument('--relations', default='impl', help='Relationship type. OPTIONS: impl, expl')
    parser.add_argument('--inFolder', default='../visualgenome/', help='Folder where visual genome files are stored')
    parser.add_argument('--outFolder', default='../training_data/', help='Where to store the output .csv files (training data)')

    args = parser.parse_args()

    # --- IMAGE META-DATA --- #
    with open(args.inFolder + 'image_data.json') as tweetfile:
        img_metadata = json.loads(tweetfile.read())
    # --- RELATIONSHIPS --- #
    with open(args.inFolder + 'relationships.json') as tweetfile:
        relationships = json.loads(tweetfile.read())

    # Explicit spatial relationships
    explicit = ['on', 'next to' , 'above', 'over', 'below','behind', 'along', 'through', 'in',
                'in front of', 'near', 'beyond', 'with', 'by', 'inside of', 'on top of', 'down', 'up',
                'beneath', 'inside', 'left', 'right', 'under', 'across from', 'underneath', 'atop',
                'across','beside', 'around', 'outside', 'next', 'against', 'at', 'between', 'front', 'aside']

    MATRIX = []
    first_row = ['img_idx', 'img_id', 'rel_id', 'subj', 'rel', 'obj', 'subj_syns', 'rel_syns', 'obj_syns',
                 'subj_ctr_x', 'subj_ctr_y', 'obj_ctr_x', 'obj_ctr_y', 'subj_sd_x', 'subj_sd_y', 'obj_sd_x', 'obj_sd_y']

    MATRIX.append(first_row)

    for i in range(0,len(img_metadata)): # Loop through ALL the images

        img_idx = i # this index corresponds to the index i+1
        img_id_actual = img_metadata[i]['image_id'] # this index is how the .jpg images are named
        width_img = img_metadata[i]['width']
        height_img = img_metadata[i]['height']

        for j in range(0, len(relationships[i]['relationships'])): # Loop over RELATIONSHIPS in the image image_id
            try:
                pred = relationships[i]['relationships'][j]['predicate'].lower()

                # PREDICATE (relationship)
                rel_id = relationships[i]['relationships'][j]['relationship_id']
                rel = str(relationships[i]['relationships'][j]['predicate'].lower())
                rel_syns = str(relationships[i]['relationships'][j]['synsets'][0])
                # SUBJECT:
                subj = str(relationships[i]['relationships'][j]['subject']['name']).replace(',', '')
                subj_syns = str(relationships[i]['relationships'][j]['subject']['synsets'][0])
                x_subj = relationships[i]['relationships'][j]['subject']['x']
                y_subj = relationships[i]['relationships'][j]['subject']['y']
                width_subj = relationships[i]['relationships'][j]['subject']['w']
                height_subj = relationships[i]['relationships'][j]['subject']['h']
                # OBJECT:
                obj = str(relationships[i]['relationships'][j]['object']['name']).replace(',', '')
                obj_syns = str(relationships[i]['relationships'][j]['object']['synsets'][0])
                x_obj = relationships[i]['relationships'][j]['object']['x']
                y_obj = relationships[i]['relationships'][j]['object']['y']
                width_obj = relationships[i]['relationships'][j]['object']['w']
                height_obj = relationships[i]['relationships'][j]['object']['h']

                # Compute the centers of the object and subject
                subj_ctr_x, subj_ctr_y, subj_sd_x, subj_sd_y, obj_ctr_x, obj_ctr_y, obj_sd_x, obj_sd_y = data_tools.compute_centers(x_subj, y_subj, width_subj, height_subj, x_obj, y_obj, width_obj, height_obj) #CENTER object and subject

                # Normalize coordinates:
                obj_ctr_x, obj_ctr_y, obj_sd_x, obj_sd_y = obj_ctr_x/width_img, obj_ctr_y/height_img, obj_sd_x/width_img, obj_sd_y/height_img #NORMALIZE object
                subj_ctr_x, subj_ctr_y, subj_sd_x, subj_sd_y = subj_ctr_x/width_img, subj_ctr_y/height_img, subj_sd_x/width_img, subj_sd_y/height_img #NORMALIZE subject

                # Append row
                new_row = [img_idx, img_id_actual, rel_id, subj, rel, obj, subj_syns, rel_syns, obj_syns, subj_ctr_x, subj_ctr_y, obj_ctr_x, obj_ctr_y, subj_sd_x, subj_sd_y, obj_sd_x, obj_sd_y] #if we were successful
                if relationships[i]['relationships'][j]['synsets'] != []: #only keep predicates that have synset
                    if (args.relations == 'expl') and pred in explicit:
                        MATRIX.append(new_row)
                    elif (args.relations == 'impl') and ((len(pred.split(" ")) == 1) and (pred not in explicit)):
                        MATRIX.append(new_row)
            except:
                pass

    # --- get STATISTICS: --- #
    print('# instances = ' + str(len(MATRIX)))
    print('# different subjects = ' + str(len( list(set([MATRIX[jj][3] for jj in range(0,len(MATRIX)) ])) ) ))
    print('# different rels  = ' +  str(len( list(set([MATRIX[jj][4] for jj in range(0,len(MATRIX)) ]) ) )))
    print('# different objects = ' + str(len( list(set([MATRIX[jj][5] for jj in range(0,len(MATRIX)) ]) ) )))

    # --- write CSV --- #
    saveDir = args.outFolder + '/TRAINING_DATA-' + args.relations + '.csv'
    write_data.matrix2csv(MATRIX, saveDir)

if __name__ == "__main__":
     main()
