import os
import uuid

import pandas as pd
import streamlit as st
from streamlit_extras.grid import grid
from streamlit_extras.stylable_container import stylable_container
from utils import (
    AVAIL_LABELED_PATHS,
    AVAIL_UNLABELED_PATHS,
    LABELED_PATH,
    ROOT_PATH,
    change_requirement,
    convert_df,
    convertToNumber,
    end_of_labeling,
    fine_tune_here,
    get_pair,
    init_important_variables,
    kickoff_finetuning,
    make_results_df,
    restart_labeling,
    rxn_to_image,
    set_current_pair,
    unlabeled_fraction,
    update_current_pair_hardest,
    update_database,
)

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
    st.session_state.tab_of_site = st.sidebar.selectbox(
        "tab_of_site",
        ["Label molecules", "Fine-tune", "Score molecules"],
        # TODO make another selection for just ranking without finetuning
        label_visibility="collapsed",
    )

    if st.session_state.tab_of_site == "Label molecules":
        considered_paths = AVAIL_UNLABELED_PATHS
    elif st.session_state.tab_of_site == "Fine-tune":
        considered_paths = AVAIL_LABELED_PATHS

    st.sidebar.title("Dataset")
    avail_datasets = [os.path.basename(path)[:-4] for path in considered_paths]
    st.session_state.selected_ds = st.sidebar.selectbox(
        "selected_ds",
        avail_datasets,
        label_visibility="collapsed",
        placeholder="No datasets...",
        on_change=change_requirement,
    )

    if st.session_state.df_update_required:
        update_database(
            considered_paths[avail_datasets.index(st.session_state.selected_ds)]
        )

    if st.session_state.number_current_pair == 0:
        unlabeled_fraction()

    ######################################################
    #################### TOP OF PAGE ##################### # noqa
    ######################################################

    col1, col2, col3 = st.columns([5, 5, 5])
    with col1:
        st.header(f"Welcome to {NAME_OF_APP} app")
    with col2:
        st.image(os.path.join(ROOT_PATH, "data", "images", "liac_logo.png"), width=200)
    with col3:
        st.text("")

    st.divider()

    st.session_state.display_theme = "light"

    if st.session_state.state_of_site == "normal":
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

        ######################################################
        ################## Label molecules ################### # noqa
        ######################################################

        # TODO add tab_of_site for fine-tuning where one chooses pretrained model and
        # labelled data and then goes to state_of_site loading
        # TODO add tab_of_site for scoring where one chooses model
        if st.session_state.tab_of_site == "Label molecules":
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

            st.session_state.molecule_pair = get_pair(
                st.session_state.number_current_pair
            )
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
                    css_styles=css_styles_green
                    if st.session_state.current_pair_hardest[
                        st.session_state.number_current_pair
                    ]
                    == 1
                    else css_styles_normal,
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
                    css_styles=css_styles_green
                    if st.session_state.current_pair_hardest[
                        st.session_state.number_current_pair
                    ]
                    == 2
                    else css_styles_normal,
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

            bot_grid2 = grid([7, 7, 7])
            bot_grid2.text("")
            bot_grid2.button(
                "Get labeled data up until now :floppy_disk:",
                on_click=make_results_df,
            )

    if st.session_state.state_of_site == "end_labeling":
        st.header("Thank you for labeling!")
        st.subheader("What do you want to do next?")
        cols = st.columns(3)
        with cols[0]:
            st.button(
                "Compile data labeled so far",
                on_click=make_results_df,
            )
        with cols[1]:
            st.button(
                "Continue labeling",
                on_click=restart_labeling,
                args=[
                    considered_paths[avail_datasets.index(st.session_state.selected_ds)]
                ],
            )
        with cols[2]:
            st.button(
                "Fine-tune model",
                on_click=kickoff_finetuning,
            )

    if st.session_state.state_of_site == "get_dataset":
        output_name = f"{st.session_state.dataset_name.split('.')[0]}_labeled.csv"
        output_path = os.path.join(LABELED_PATH, output_name)
        df = pd.read_csv(output_path)
        csv = convert_df(df)
        cols = st.columns(3)
        with cols[0]:
            st.text("")
        with cols[1]:
            st.header("Click below to download your labeled data")
            st.download_button(
                "Download data as CSV",
                file_name=output_name,
                data=csv,
                use_container_width=True,
            )

    if st.session_state.state_of_site == "loading":
        cols = st.columns(3)
        with cols[0]:
            st.text("")
        with cols[1]:
            st.header("Please wait, your model is being fine tuned...")
        fine_tune_here()
        st.session_state.state_of_site = "give_model"
        st.experimental_rerun()

    if st.session_state.state_of_site == "give_model":
        cols = st.columns(3)
        with cols[0]:
            st.text("")
        with cols[1]:
            st.header("Click below to download your model")
            st.download_button(
                "Your fine-tuned model",
                file_name="fine_tuned_model.pt",
                data=b"fake_model",
                use_container_width=True,
            )  # !!! Change here !!!


######################################################
################### BOTTOM OF PAGE ################### # noqa
######################################################


if __name__ == "__main__":
    main()