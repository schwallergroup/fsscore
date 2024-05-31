import glob
import os
import uuid

import pandas as pd
import streamlit as st
from streamlit_extras.grid import grid
from streamlit_extras.stylable_container import stylable_container
from utils import (
    LABELED_PATH,
    MODELS_PATH,
    ROOT_PATH,
    SCORED_PATH,
    UNLABELED_PATH,
    _go_to_labeling,
    convert_df,
    convertToNumber,
    end_of_labeling,
    fine_tune,
    get_pair,
    go_to_labeling,
    go_to_loading,
    go_to_score_loading,
    init_important_variables,
    kickoff_finetuning,
    pair_molecules,
    rank_by_uncertainty,
    restart_labeling,
    rxn_to_image,
    score,
    set_current_pair,
    update_current_pair_hardest,
    update_database,
)

from fsscore.utils.paths import PRETRAIN_MODEL_PATH

NAME_OF_APP = "FSscore"

max_number_rxn_shown_top = 7  # Must be odd for better visualisation
number_rxn_shown_top_each_side = max_number_rxn_shown_top // 2

if "session_name" not in st.session_state:
    st.session_state["session_name"] = str(uuid.uuid4())
    st.session_state.random_state_dataset = (
        convertToNumber(st.session_state.session_name)
    ) % (2**30)

st.set_page_config(
    page_title=f"{NAME_OF_APP} · LIAC", page_icon=":alembic:", layout="wide"
)

css_styles_green = """
button{
    background-color: green;
    color: white;
    border-radius: 20px;
}
"""

css_styles_normal = """"""

# hide streamlit menu and change footer
hide_streamlit_style = """
            <style>
            #MainMenu {visibility: hidden;}
            footer {visibility: hidden;}
            footer:after {
                content:'Made by Théo Neukomm and Rebecca Neeser';
                font-weight: bold;
                visibility: visible;
                display: block;
                position: relative;
                #background-color: red;
                padding: 5px;
                top: 2px;
            }
            </style>
            """
st.markdown(hide_streamlit_style, unsafe_allow_html=True)


def main():
    init_important_variables()

    ######################################################
    ###################### Sidebar ####################### # noqa
    ######################################################
    st.sidebar.title("Usage")
    options = ["Home", "Label molecules", "Fine-tune", "Score molecules"]
    index = options.index(st.session_state.tab_of_site)

    st.session_state.tab_of_site = st.sidebar.selectbox(
        "tab_of_site",
        options,
        label_visibility="collapsed",
        index=index,
        on_change=lambda: st.session_state.state_of_site == "normal",
    )

    if st.session_state.tab_of_site == "Label molecules":
        considered_paths = []
    elif st.session_state.tab_of_site == "Fine-tune":
        considered_paths = glob.glob(os.path.join(LABELED_PATH, "*.csv"))
    elif st.session_state.tab_of_site == "Score molecules":
        considered_paths = glob.glob(
            os.path.join(ROOT_PATH, "data", "scoring", "*.csv")
        )
    elif st.session_state.tab_of_site == "Home":
        considered_paths = []
        st.session_state.state_of_site = "normal"

    st.sidebar.title("Dataset")
    avail_datasets = [os.path.basename(path)[:-4] for path in considered_paths]

    st.session_state.selected_ds = st.sidebar.selectbox(
        "selected_ds",
        avail_datasets,
        label_visibility="collapsed",
        placeholder="No datasets...",
        # on_change=reset_position,
    )

    if avail_datasets:
        update_database(
            considered_paths[avail_datasets.index(st.session_state.selected_ds)]
        )

    ######################################################
    #################### TOP OF PAGE ##################### # noqa
    ######################################################

    col1, col2, col3 = st.columns([5, 5, 5])
    with col1:
        st.header(f"{NAME_OF_APP} app")
    with col2:
        st.image(os.path.join(ROOT_PATH, "data", "images", "liac_logo.png"), width=200)
    with col3:
        st.text("")

    st.divider()

    st.session_state.display_theme = "light"

    if st.session_state.state_of_site == "normal":
        if st.session_state.tab_of_site == "Home":
            st.header("Welcome to FSscore!")
            st.text("This app allows you to label molecules,")
            st.text("fine-tune a model based on those labels")
            st.text(" and finally score new molecules.")
            st.subheader("Please select a tab on the left to start.")
            st.text("Select the 'Label molecules' tab for a new project.")

        if st.session_state.tab_of_site == "Label molecules":
            # intro to labeling
            # allows uploading of new dataset and pairing of molecules
            st.header("Label molecules")

            grid1 = grid([7, 3, 7])
            grid1.subheader("Upload a dataset to start labeling")
            grid1.subheader("OR")
            grid1.subheader("Select a paired dataset to continue labeling")

            grid2 = grid([7, 3, 7])
            grid2.text(
                "The dataset will be split into pairs of molecules. \nOnly csv files with a 'smiles' column are accepted. \nThis is dependent on the pre-trained model from the publication. \nMake sure you have that in the 'models' folder."  # noqa
            )
            grid2.text("")
            grid2.text("Select a dataset from the dropdown menu.")

            # get upload file
            grid3 = grid([7, 3, 7])
            st.session_state.uploaded_file = grid3.file_uploader(
                "Upload a csv file",
                type=["csv"],
            )

            if st.session_state.uploaded_file is not None:
                st.session_state.state_of_site = "upload_pair"
            grid3.text("")
            labeling_data = glob.glob(os.path.join(UNLABELED_PATH, "*.csv"))
            avail_datasets = [os.path.basename(path)[:-4] for path in labeling_data]
            st.session_state.paired_ds = grid3.selectbox(
                "Make a selection:",
                avail_datasets,
                index=None,
                label_visibility="collapsed",
                placeholder="No datasets...",
            )

            grid4 = grid([7, 3, 7])
            grid4.text("")
            grid4.text("")
            grid4.button("Continue", on_click=_go_to_labeling)

        if st.session_state.tab_of_site == "Fine-tune":
            st.header("Fine-tune your model")

            cols = st.columns(2)
            with cols[0]:
                st.subheader("1) Dataset")
                st.text("Select the dataset to fine-tune on to the left.")
                st.text("")
                st.subheader("2) Pre-trained model")
                st.text("Select the graph-based model you want to fine-tune")
                st.text("Fingerprint-based models are not supported in the app.")
                st.session_state.model_path = st.text_input(
                    "Insert path to the model or keep default.",
                    value=PRETRAIN_MODEL_PATH,
                )

            with cols[1]:
                # sliders for hyperparameters
                st.subheader("3) Select hyperparameters:")
                st.text("Default values are set for the hyperparameters.")
                st.session_state.num_samples = st.slider(
                    "Number of samples (-1: all labeled samples)", -1, 100, -1
                )  # noqa
                st.session_state.lr = st.slider("Learning rate", 0.0001, 0.01, 0.003)
                st.session_state.batch_size = st.slider("Batch size", 4, 128, 4)
                st.session_state.num_epochs = st.slider("Number of epochs", 1, 100, 20)
                st.session_state.patience = st.slider(
                    "Patience for earlz stopping", 1, 10, 3
                )

            st.divider()

            # fine-tune button
            option_grid2 = grid([1, 5, 1])
            option_grid2.text("")
            option_grid2.button(
                "Start fine-tuning",
                on_click=go_to_loading,
            )
            option_grid2.text("")

        if st.session_state.tab_of_site == "Score molecules":
            # not implemented yet
            st.header("Score molecules")
            st.subheader("1) Data")
            st.text("Select the dataset to score on to the left.")
            st.text("")
            st.subheader("2) Model")
            st.text("Select the model to score with.")
            # give drop-down menu for model with selectbox
            avail_models = glob.glob(os.path.join(MODELS_PATH, "*.ckpt"))
            avail_models = [os.path.basename(path) for path in avail_models]
            st.session_state.model_path_scoring = st.selectbox(
                "model_path_scoring",
                avail_models,
                index=None,
                label_visibility="collapsed",
                placeholder="No models...",
            )
            if st.session_state.model_path_scoring:
                st.session_state.model_path_scoring = os.path.join(
                    MODELS_PATH, st.session_state.model_path_scoring
                )
            st.divider()

            # score button
            option_grid2 = grid([1, 5, 1])
            option_grid2.text("")
            option_grid2.button(
                "Score",
                on_click=go_to_score_loading,
            )
            option_grid2.text("")

    if st.session_state.state_of_site == "_labeling":
        st.header("Label molecules")
        cols = st.columns(int(max_number_rxn_shown_top))

        for i in range(max_number_rxn_shown_top):
            i_shift = i - number_rxn_shown_top_each_side
            i_rxn = st.session_state.number_current_pair + i_shift + 1
            if i_rxn > 0 and i_rxn <= st.session_state.number_pair_feedback:
                with cols[i]:
                    if i_shift == 0:
                        st.success(i_rxn)
                    else:
                        st.text(i_rxn)
            else:
                with cols[i]:
                    st.text("")

        st.session_state.loading_bar = st.progress(0, text="")
        if (
            st.session_state.number_pair_feedback != 0
            and not st.session_state.final_validation
        ):
            st.session_state.loading_bar.progress(
                (st.session_state.number_current_pair)
                / st.session_state.number_pair_feedback
            )
        elif (
            st.session_state.number_pair_feedback != 0
            and st.session_state.final_validation
        ):
            st.session_state.loading_bar.progress(1.0)
        st.divider()

        st.session_state.molecule_pair = get_pair(st.session_state.number_current_pair)
        if (
            st.session_state.pair_number_last_images
            != st.session_state.number_current_pair
        ):
            (
                st.session_state.image_mol_1,
                st.session_state.image_mol_2,
            ) = rxn_to_image(
                st.session_state.molecule_pair[0], st.session_state.display_theme
            ), rxn_to_image(
                st.session_state.molecule_pair[1], st.session_state.display_theme
            )

        main_grid = grid(5, 5, [1, 4], 5, vertical_align="bottom")

        main_grid.text("")
        main_grid.header("Molecule 1")
        main_grid.text("")
        main_grid.header("Molecule 2")
        main_grid.text("")

        main_grid.text("")
        main_grid.image(st.session_state.image_mol_1)
        main_grid.text("")
        main_grid.image(st.session_state.image_mol_2)
        main_grid.text("")

        main_grid.text("")
        main_grid.subheader("Which molecule is harder to synthesize?")

        main_grid.text("")
        with main_grid.container():
            with stylable_container(
                key="button_mol_1",
                css_styles=(
                    css_styles_green
                    if st.session_state.current_pair_hardest[
                        st.session_state.number_current_pair
                    ]
                    == 1
                    else css_styles_normal
                ),
            ):
                st.button(
                    "Molecule 1 :arrow_up:",
                    on_click=update_current_pair_hardest,
                    args=[1],
                )
        main_grid.text("")
        with main_grid.container():
            with stylable_container(
                key="button_mol_2",
                css_styles=(
                    css_styles_green
                    if st.session_state.current_pair_hardest[
                        st.session_state.number_current_pair
                    ]
                    == 2
                    else css_styles_normal
                ),
            ):
                st.button(
                    "Molecule 2 :arrow_up:",
                    on_click=update_current_pair_hardest,
                    args=[2],
                )
        main_grid.text("")
        st.divider()

        bot_grid = grid([1, 5, 2, 5, 2, 5, 1])

        bot_grid.text("")
        if st.session_state.number_current_pair > 0:
            bot_grid.button(
                ":arrow_left: Previous pair",
                on_click=set_current_pair,
                args=[st.session_state.number_current_pair - 1],
            )
        else:
            bot_grid.text("")
        bot_grid.text("")

        if (
            st.session_state.number_current_pair + 1
            < st.session_state.number_pair_feedback
        ):
            bot_grid.button("Send labels :envelope:", on_click=end_of_labeling)
        bot_grid.text("")

        if (
            st.session_state.number_current_pair + 1
            < st.session_state.number_pair_feedback
        ):
            bot_grid.button(
                "Next pair :arrow_right:",
                on_click=set_current_pair,
                args=[st.session_state.number_current_pair + 1],
            )
        elif (
            st.session_state.number_current_pair + 1
            == st.session_state.number_pair_feedback
            and st.session_state.number_pair_feedback != 0
        ):
            bot_grid.button("Send labels :envelope:", on_click=end_of_labeling)
        else:
            bot_grid.text("")

    if st.session_state.state_of_site == "end_labeling":
        st.header("Thank you for labeling!")
        st.subheader("What do you want to do next?")
        cols = st.columns(3)
        with cols[0]:
            output_name = f"{st.session_state.dataset_name.split('.')[0]}_labeled.csv"
            output_path = os.path.join(LABELED_PATH, output_name)
            df = pd.read_csv(output_path)
            csv = convert_df(df)
            st.download_button(
                "Download labeled data",
                file_name=output_name,
                data=csv,
                use_container_width=True,
            )
        with cols[1]:
            st.button(
                "Continue labeling",
                on_click=restart_labeling,
                args=[os.path.join(UNLABELED_PATH, st.session_state.dataset_name)],
            )
        with cols[2]:
            st.button(
                "Fine-tune model",
                on_click=kickoff_finetuning,
            )

    if st.session_state.state_of_site == "upload_pair":
        st.markdown("##")
        cols = st.columns(3)
        with cols[0]:
            st.text("")
        with cols[1]:
            with st.spinner("Clustering and pairing molecules..."):
                df_pairs = pair_molecules(st.session_state.uploaded_file)

        # new spinner for rearranging pairs
        output_path = os.path.join(
            UNLABELED_PATH,
            f"{os.path.basename(st.session_state.uploaded_file.name).split('.')[0]}_pairs.csv",  # noqa
        )
        with cols[1]:
            with st.spinner("Ranking based on prediction uncertainty..."):
                rank_by_uncertainty(df_pairs, output_path)

        st.session_state.number_pair_feedback = min(len(df_pairs), 100)
        st.session_state.number_current_pair = 0
        st.session_state.final_validation = False
        st.session_state.current_pair_hardest = [0] * len(df_pairs)
        st.session_state.pair_number_last_images = -1

        st.button("Continue with labeling", on_click=go_to_labeling, args=[output_path])

    if st.session_state.state_of_site == "loading":
        # insert some vertical space
        st.markdown("##")
        cols = st.columns(3)
        with cols[0]:
            st.text("")
        with cols[1]:
            with st.spinner("Fine-tuning model..."):
                fine_tune()
        st.session_state.state_of_site = "give_model"

    if st.session_state.state_of_site == "give_model":
        st.header("Model fine-tuned!")
        st.subheader("The checkpoint is saved at:")
        st.text(st.session_state.ft_model_path)

    if st.session_state.state_of_site == "loading_scoring":
        # insert some vertical space
        st.markdown("##")
        cols = st.columns(3)
        with cols[0]:
            st.text("")
        with cols[1]:
            with st.spinner("Scoring molecules..."):
                score()
        st.session_state.state_of_site = "Molecules scored!"
        st.subheader("The results are saved in folder:")
        st.text(SCORED_PATH)


######################################################
################### BOTTOM OF PAGE ################### # noqa
######################################################


if __name__ == "__main__":
    main()
