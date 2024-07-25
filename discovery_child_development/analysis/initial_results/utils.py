import pandas as pd
import json
import ast

from discovery_child_development.utils import plotting_utils as pu
from discovery_child_development.utils import analysis_utils as au
from discovery_child_development.utils import openalex_utils
from discovery_child_development.getters import patents
from discovery_child_development import PROJECT_DIR

ENRICHED_DATA_DIR = PROJECT_DIR / "outputs/enrichments"
INPUTS_DATA_DIR = PROJECT_DIR / "inputs/data"
PATH_TO_TOPICS = (
    PROJECT_DIR
    / "discovery_child_development/pipeline/labelling/taxonomy_cat/prompts/topics.json"
)

EARLY_STAGE_DEALS = [
    "angel",
    "convertible_note",
    "equity_crowdfunding",
    "non_equity_assistance",
    "pre_seed",
    "product_crowdfunding",
    "secondary_market",
    "seed",
    "series_a",
    "series_b",
    "series_c",
    "series_d",
    "series_e",
    "series_unknown",
]


def load_ukri_data():
    metadata_df = pd.read_csv(ENRICHED_DATA_DIR / "gtr_texts.csv")
    data_df = (
        pd.read_csv(
            ENRICHED_DATA_DIR / "taxonomy_cat/taxonomy_cat_predictions_gtr_filtered.csv"
        )
        .assign(dataset="UKRI")
        .assign(year=lambda df: df["start"].apply(lambda x: int(x.split("-")[0])))
        .drop(columns=["start", "end"])
        .merge(metadata_df[["id", "amount"]], on="id", how="left")
        .assign(amount=lambda df: df.amount / 1000)
        .query("year >= 2013 and year <= 2023")
    )
    return data_df


def get_baseline_ukri(df: pd.DataFrame = None):
    if df is None:
        df = pd.read_csv(ENRICHED_DATA_DIR / "gtr_texts.csv")
    return (
        df.assign(
            year=lambda df: pd.to_datetime(df["start"]).dt.year,
            amount=lambda df: df["amount"] / 1000,
        )
        .groupby("year")
        .agg(
            counts=("id", "nunique"),
            amount=("amount", "sum"),
        )
        .reset_index()
        .query("year >= 2013 and year <= 2023")
    )


def load_openalex_data():
    data_df = (
        pd.read_csv(
            ENRICHED_DATA_DIR
            / "taxonomy_cat/taxonomy_cat_predictions_openalex_filtered_final.csv"
        )
        .assign(id=lambda df: df["id"].apply(lambda x: x.split("/")[-1]))
        .query("source == 'openalex'")
        .rename(columns={"source": "dataset"})
    )
    data_df_manual = pd.read_csv(
        ENRICHED_DATA_DIR / "openalex_manual_additions.csv"
    ).assign(
        country_code=lambda df: df["country_code"].apply(lambda x: ast.literal_eval(x))
    )
    metadata_df = (
        pd.read_csv(ENRICHED_DATA_DIR / "openalex_metadata_df_final.csv")
        .drop_duplicates(subset=["id"])
        .dropna(subset=["country_code"])
        .assign(
            country_code=lambda df: df["country_code"].apply(
                lambda x: ast.literal_eval(x)
            )
        )
    )
    return pd.concat(
        [
            (
                data_df.merge(
                    metadata_df[["id", "year", "country_code"]], how="left", on="id"
                ).query("year >= 2013 and year <= 2023")
            ),
            data_df_manual,
        ],
        ignore_index=True,
    )


# def _load_openalex_data():
#     data_df = (
#         pd.read_csv(
#             ENRICHED_DATA_DIR / "taxonomy_cat/taxonomy_cat_predictions_filtered.csv"
#         )
#         .assign(id=lambda df: df["id"].apply(lambda x: x.split("/")[-1]))
#         .query("source == 'openalex'")
#         .rename(columns={"source": "dataset"})
#     )
#     metadata_df = (
#         pd.read_csv(ENRICHED_DATA_DIR / "openalex_concepts_metadata.csv")
#         .rename(columns={"openalex_id": "id"})
#         .assign(id=lambda df: df["id"].apply(lambda x: x.split("/")[-1]))
#         .drop_duplicates(subset=["id"])
#     )
#     extra_metadata_df = pd.read_csv(ENRICHED_DATA_DIR / "pubs_metadata_df.csv").assign(
#         country_code=lambda df: df["country_code"].apply(lambda x: ast.literal_eval(x))
#     )
#     return (
#         data_df.merge(metadata_df[["id", "year"]], how="left", on="id")
#         .merge(extra_metadata_df, how="left", on="id")
#         .query("year >= 2013 and year <= 2023")
#     )


def get_baseline_openalex():
    return openalex_utils.get_publications_count_per_year(
        start_year=2013, end_year=2024
    ).query("year >= 2013 and year <= 2023")


def load_patents_data():
    data_df = (
        pd.read_csv(
            ENRICHED_DATA_DIR
            / "taxonomy_cat/taxonomy_cat_predictions_patents_filtered.csv"
        )
        .query("source == 'patents'")
        .rename(columns={"source": "dataset"})
    )
    metadata_df = (
        patents.get_patents_from_s3()
        .rename(columns={"publication_number": "id"})
        .drop_duplicates(subset=["id"])
        .assign(
            year=lambda df: df["publication_date"].apply(lambda x: int(str(x)[0:4]))
        )
    )[["id", "year", "country_code"]]
    return data_df.merge(metadata_df, on="id", how="left").query(
        "year >= 2013 and year <= 2023"
    )


def get_baseline_patents(df: pd.DataFrame = None):
    if df is None:
        df = pd.DataFrame(json.load(open("total_patents.json", "r")))
    return (
        df.rename(columns={"total_publications": "counts", "publication_year": "year"})
        .astype({"year": int, "counts": int})
        .query("year >= 2013 and year <= 2023")
    )


def load_crunchbase_companies():
    return pd.read_csv(ENRICHED_DATA_DIR / "crunchbase_combined_labels_checked.csv")


def load_crunchbase_data():
    cb_data_df = pd.read_csv(
        ENRICHED_DATA_DIR / "crunchbase_combined_labels_checked.csv"
    )
    # add missing companies - Byju's
    df_extra = pd.DataFrame(
        {
            "id": ["15d119e6-d721-3baf-da4b-880891c0c3fd"],
            "topics": ["mobile, literacy, numeracy, internet"],
        }
    )
    cb_data_df = pd.concat([cb_data_df, df_extra], ignore_index=True)

    cb_country_codes = (
        pd.read_csv(ENRICHED_DATA_DIR / "crunchbase_country_codes.csv")
        .drop_duplicates(subset=["id"])
        .rename(columns={"id": "org_id"})
    )
    return (
        pd.read_parquet(INPUTS_DATA_DIR / "crunchbase/funding_rounds_full.parquet")
        .drop_duplicates("funding_round_id")
        .query("org_id in @cb_data_df['id'].unique()")
        .assign(year=lambda df: df.announced_on.apply(lambda x: int(x[:4])))
        .query("year >= 2013 and year <= 2023")
        .merge(
            cb_data_df[["id", "text", "topics"]],
            left_on="org_id",
            right_on="id",
            how="left",
        )
        .rename(columns={"raised_amount_gbp": "amount"})
        .assign(dataset="crunchbase")
        .assign(id=lambda df: df["funding_round_id"])
        .query("investment_type in @EARLY_STAGE_DEALS")
        .drop("country_code", axis=1)
        .merge(cb_country_codes, on="org_id", how="left")
    )[
        [
            "id",
            "text",
            "dataset",
            "topics",
            "year",
            "country_code",
            "amount",
            "investment_type",
            "org_id",
        ]
    ]


def get_baseline_crunchbase():
    return (
        pd.read_parquet(INPUTS_DATA_DIR / "crunchbase/funding_rounds_full.parquet")
        .drop_duplicates("funding_round_id")
        .assign(year=lambda df: df.announced_on.apply(lambda x: int(x[:4])))
        .query("year >= 2013 and year <= 2023")
        .rename(columns={"raised_amount_gbp": "amount"})
        .assign(id=lambda df: df["funding_round_id"])
        .query("investment_type in @EARLY_STAGE_DEALS")
        .groupby("year")
        .agg(
            counts=("id", "nunique"),
            amount=("amount", "sum"),
        )
        .reset_index()
    )


def adjust_by_uk_inflation(df, column="amount", reference_year=2019):
    inflation_df = pd.read_csv(INPUTS_DATA_DIR / "gdp_deflator.csv").query(
        "year >= 2013 and year <= 2023"
    )
    target_deflator = inflation_df.query("year == @reference_year")["deflator"].values[
        0
    ]
    return (
        df.merge(inflation_df[["year", "deflator"]], on="year", how="left")
        .assign(real_terms=lambda df: df[column] / df["deflator"] * target_deflator)
        .drop(columns=["deflator"])
    )


def get_geographical_distribution(data_exploded_df, column="id"):
    data_countries_df = data_exploded_df.dropna(
        subset=["country_code"]
    ).drop_duplicates(["id", "country_code"])
    country_codes = data_countries_df.country_code.unique()

    growth_df = []
    ts_counts = []
    for country_code in country_codes:
        country_df = data_countries_df.query("country_code == @country_code").drop(columns=["country_code"])
        _ts_df = get_timeseries(country_df, column=column)
        growth_df.append(
            au.ts_magnitude_growth_(ts_df=_ts_df, year_start=2019, year_end=2023)
            .assign(country_code=country_code)
            .reset_index(drop=True)
        )
        ts_counts.append(_ts_df.assign(country_code=country_code))
    growth_df = pd.concat(growth_df, ignore_index=True)
    ts_counts = pd.concat(ts_counts, ignore_index=True)
    return growth_df, ts_counts


def load_topic_data(is_crunchbase=False):
    topics_dict = json.load(open(PATH_TO_TOPICS, "r"))
    topics = list(topics_dict.keys())

    topics_df = []
    for topic in topics_dict:
        topics_df.append(
            {
                "topic": topic,
                "type": topics_dict[topic]["type"],
                "subtype": topics_dict[topic]["subtype"],
                "name": topics_dict[topic]["name"],
            }
        )
    topics_df = (
        pd.DataFrame(topics_df)
        .sort_values(
            [
                "type",
                "subtype",
                "topic",
            ]
        )
        .reset_index(drop=True)
        .replace("Family and home", "Parenting")
        .rename(columns={"type": "type"})
        .replace("Data science and AI", "AI")
    )
    if is_crunchbase:
        return pd.concat(
            [
                topics_df,
                pd.DataFrame(
                    {
                        "topic": ["operations"],
                        "type": ["Technology"],
                        "subtype": ["Operations"],
                        "name": ["Operations"],
                    }
                ),
            ],
            ignore_index=True,
        )
    else:
        return topics_df


TOPICS_DF = load_topic_data()


def count_topic_mentions(data_df, column="topics"):
    return data_df.groupby(column).agg(counts=("id", "count")).reset_index()


def plot_quick_ts(data_df, column, value_denominator=1):
    fig = pu.ts_smooth(
        (
            data_df.assign(Total="Total").assign(
                amount=lambda df: df[column] / value_denominator
            )
        ),
        ["Total"],
        variable="amount",
        variable_title="",
        category_column="Total",
        width=300,
        height=150,
        legend_orient="right",
    )
    return pu.configure_plots(fig)


def explode_data(data_df, column="topics", is_crunchbase=False):
    return (
        data_df.fillna({column: ","})
        .assign(
            topics=lambda df: df.topics.apply(
                lambda x: [x.strip() for x in x.split(",")]
            )
        )
        .explode("topics")
        .merge(
            load_topic_data(is_crunchbase),
            left_on="topics",
            right_on="topic",
            how="left",
            suffixes=("", "_"),
        )
        .drop_duplicates(subset=["id", "topics", "type"])
    )


def get_timeseries(data_df, groupby=["year"], column="id"):
    if column == "id":
        column_name = "counts"
        agg = "count"
    elif column == "amount":
        column_name = column
        agg = "sum"

    ts_df = (
        data_df.groupby(groupby)
        .agg({column: agg})
        .reset_index()
        .rename(columns={column: column_name})
        .query("year >= 2013 and year <= 2023")
    )
    return au.impute_empty_periods(
        # convert year column to datetime
        ts_df.assign(year=lambda df: pd.to_datetime(df.year, format="%Y")),
        "year",
        "Y",
        2013,
        2023,
    ).assign(year=lambda df: df.year.dt.year)


def impute_empty_periods_all_ts(ts_df, cat_column):
    dfs = []
    for cat in ts_df[cat_column].unique():
        dfs.append(
            au.impute_empty_periods(
                ts_df.query(f"{cat_column} == '{cat}'").assign(
                    year=lambda df: pd.to_datetime(df.year, format="%Y")
                ),
                "year",
                "Y",
                2013,
                2023,
            ).assign(**{cat_column: cat})
        )
    return pd.concat(dfs, ignore_index=True).assign(year=lambda df: df.year.dt.year)


def magnitude_and_growth(ts_df, column, value):
    cats = ts_df[column].unique()
    dfs = []
    for cat in cats:
        dfs.append(
            au.ts_magnitude_growth_(
                ts_df.query(f"{column} == @cat")[["year", value]],
                year_start=2019,
                year_end=2023,
            )
            .assign(**{column: cat})
            .reset_index(drop=True)
        )
    dfs = pd.concat(dfs)
    return dfs


def get_data_distribution(data_exploded_df, column, values, ts: bool = False):
    dfs = []
    for value in values:
        if value == "id":
            column_name = "counts"
            agg = "nunique"
            normalisation_value = len(data_exploded_df.drop_duplicates("id"))
        elif value == "amount":
            column_name = value
            agg = "sum"
            normalisation_value = data_exploded_df.drop_duplicates("id")["amount"].sum()

        if ts:
            groupby = ["year", column]
        else:
            groupby = [column]

        _df = (
            data_exploded_df.drop_duplicates(["id", column])
            .groupby(groupby)
            .agg(**{column_name: (value, agg)})
            .reset_index()
            .assign(
                **{
                    f"{column_name}_prop": lambda df: round(
                        df[column_name] / normalisation_value, 3
                    )
                }
            )
        )
        if ts:
            _df = _df.pipe(impute_empty_periods_all_ts, column)
        dfs.append(_df)

    df = pd.concat(dfs, axis=1).T.drop_duplicates().T
    if column == "type":
        return df
    if column == "subtype":
        return df.merge(
            TOPICS_DF[["type", "subtype"]].drop_duplicates(), on="subtype", how="left"
        )


def get_data_magnitude_growth(data_exploded_df, ids, column, value):
    if ids is None:
        ids = data_exploded_df["id"].unique()

    tech_applications_df = get_data_distribution(
        data_exploded_df.query("id in @ids").query("year >= 2019"),
        column=column,
        values=["id"],
        ts=False,
    )
    tech_applications_ts = get_data_distribution(
        data_exploded_df.query("id in @ids"), column=column, values=[value], ts=True
    )

    _value = "counts" if value == "id" else value

    df = (
        magnitude_and_growth(tech_applications_ts, column, _value)
        .merge(tech_applications_df[[column, "counts"]], on=column, how="left")
        .sort_values("growth", ascending=False)
    )
    if column == "type":
        return df
    if column == "subtype":
        return df.merge(
            TOPICS_DF[["type", "subtype"]].drop_duplicates(), on="subtype", how="left"
        )


show_types = [
    "Biosciences",
    "Child care & preschool",
    "Development & learning",
    "Health",
    "Society",
    "Parenting",
]


def _get_counts_by_application(
    data_exploded_df,
    topics_df,
    selected_ids,
    groupby_column="name",
    count_col="id",
    count_agg="count",
):
    return (
        data_exploded_df.query("year >= 2019")
        .query("id in @selected_ids")
        .drop_duplicates(["id", "name"])
        .groupby(groupby_column)
        .agg(counts=(count_col, count_agg))
        .reset_index()
        .merge(topics_df, on=groupby_column, how="left")
        .sort_values(["type", "counts"], ascending=[True, False])
        .query("type in @show_types")
    )[["topic", "name", "subtype", "type", "counts"]]


def get_counts_by_application_all_tech(
    data_exploded_df, topics_df, count_col="id", count_agg="count"
):
    counts = []
    _df = topics_df.query("type in @show_types").query("topic != 'arts'")[["name"]]
    for tech_topic in ["AI", "Internet", "Mobile", "Immersive tech"]:
        selected_ids = data_exploded_df.query("subtype == @tech_topic").id.to_list()
        counts_df = _get_counts_by_application(
            data_exploded_df,
            topics_df,
            selected_ids,
            count_col=count_col,
            count_agg=count_agg,
            groupby_column="name",
        )[["counts", "name"]].rename(columns={"counts": tech_topic})
        _df = _df.merge(counts_df, on="name", how="left")
    _df = _df.fillna(0)
    return _df


def get_counts_by_application(
    data_exploded_df, topics_df, count_col="id", count_agg="count"
):
    selected_ids = data_exploded_df.query("type == 'Technology'").id.to_list()
    _total_counts = _get_counts_by_application(
        data_exploded_df,
        topics_df,
        selected_ids,
        count_col=count_col,
        count_agg=count_agg,
        groupby_column="name",
    )

    _tech_counts = get_counts_by_application_all_tech(
        data_exploded_df, topics_df, count_col=count_col, count_agg=count_agg
    )

    return (
        _total_counts[["name", "counts"]]
        .merge(_tech_counts, on="name")
        .rename(columns={"counts": "Total"})
    )


import altair as alt


def get_counts_by_application_chart(
    df,
    chart_title="Digital technology applications (detailed)",
    chart_subtitle="Number of publications",
):
    fig = (
        alt.Chart(
            df,
            width=300,
            height=400,
        )
        .mark_bar()
        .encode(
            y=alt.Y("name:N", sort=df.name.to_list(), title=""),
            x=alt.X("counts:Q", title=""),
            color=alt.Color("type:N", legend=alt.Legend(title="Type")),
            tooltip=["name", "counts"],
        )
    )

    fig = pu.configure_titles(pu.configure_plots(fig), chart_title, chart_subtitle)
    return fig
