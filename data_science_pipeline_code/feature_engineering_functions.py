def get_previous_days_count(df, hr):
    hour_df = df[df['hr'] == hr]
    sorted_df = hour_df.sort_values(by='dt')
    sorted_df['previous_week_count'] = (sorted_df['cnt'].diff(periods=7) - sorted_df['cnt']) * -1
    output = sorted_df.dropna()
    return output
