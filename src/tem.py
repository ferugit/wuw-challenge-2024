import statistics


def calculate_tem(hypothesis_df, reference_df, collar=0.0):
    """
    This funtion calculates the Time-Error Metric (TEM) generated alignments and reference alignments.
    """
    
    # Calculate the time error metric
    Tini=[]
    Tfin=[]
    TE=[]

    # Filter samples that contain the Label = "WuW" or "WuW+Command"
    reference_df = reference_df[reference_df['Label'].isin(['WuW', 'WuW+Command'])]
    false_negatives = 0

    for idx, row in reference_df.iterrows():

        row_hypothesis = hypothesis_df[hypothesis_df['Filename'] == row['Filename']]

        if row_hypothesis.empty:
            raise Exception(f"File {row['Filename']} not found in the hypothesis file")
        
        # Check if the hypothesis file has valid values (!= "Unknown")
        if row_hypothesis['Start_Time'].values[0] == 'Unknown':
            false_negatives += 1
            continue
        
        ref = [float(row['Original_Audio_Onset']), float(row['Original_Audio_Onset']) + float(row['Original_Audio_Length'])]
        hyp = [float(row_hypothesis['Start_Time'].values[0]), float(row_hypothesis['End_Time'].values[0])]
        
        ini=abs(float(ref[0])-float(hyp[0]))
        
        if (ini<collar):
            ini=0.0
        
        fin=abs(float(ref[1])-float(hyp[1]))
       
        if (fin<collar):
            fin=0.0
        
        Tini.append(ini)
        Tfin.append(fin)
        TE.append(ini+fin)

    return statistics.median(TE)
