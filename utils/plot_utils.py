
def get_s(df, x_name, y_name):
    df_temp = df.groupby([x_name, y_name, "label"]).size().reset_index().rename(columns={0: 'count'})
    s = [df_temp.loc[(df_temp[x_name] == x) & (df_temp[y_name] == y) & (df_temp["label"] == z)]["count"].tolist() for
         x, y, z in zip(df[x_name], df[y_name], df["label"])]
    s = [item + 2 for sublist in s for item in sublist]  # +2 to make the markers larger
    return s
