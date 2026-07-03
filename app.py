import streamlit as st
from PIL import Image
import numpy as np


# ----------------------------------------------------
# Page Configuration
# ----------------------------------------------------

st.set_page_config(
    page_title="Face Detection & Recognition",
    page_icon="😊",
    layout="wide",
    initial_sidebar_state="expanded"
)



from utils import (
    process_detection,
    verify_faces
)



# ----------------------------------------------------
# Sidebar
# ----------------------------------------------------

st.sidebar.title("😊 Face Detection & Recognition")

st.sidebar.markdown("---")



page = st.sidebar.radio(
    "Choose Module",
    [
      
        "Face Detection",
        "Face Verification"
    ]
)



# ----------------------------------------------------
# HOME
# ----------------------------------------------------



# ----------------------------------------------------
# FACE DETECTION
# ----------------------------------------------------

if page=="Face Detection":

    st.title("📷 Face Detection")

    uploaded=st.file_uploader(
        "Upload Image",
        type=["jpg","jpeg","png"]
    )

    if uploaded:

        image=Image.open(uploaded).convert("RGB")
        image=np.array(image)

        with st.spinner("Detecting Faces..."):

            output,total_faces=process_detection(image)

        st.image(
            output,
            use_container_width=True
        )

        st.success(f"Total Faces Detected : {total_faces}")

# ----------------------------------------------------
# FACE VERIFICATION
# ----------------------------------------------------

elif page=="Face Verification":

    st.title("👤 Face Verification")

    col1,col2=st.columns(2)

    with col1:

        img1=st.file_uploader(
            "Upload Reference Image",
            type=["jpg","jpeg","png"],
            key="1"
        )

    with col2:

        img2=st.file_uploader(
            "Upload Test Image",
            type=["jpg","jpeg","png"],
            key="2"
        )

    if img1 and img2:

        image1=np.array(Image.open(img1).convert("RGB"))
        image2=np.array(Image.open(img2).convert("RGB"))

        c1,c2=st.columns(2)

        with c1:

            st.image(
                image1,
                caption="Reference Image",
                use_container_width=True
            )

        with c2:

            st.image(
                image2,
                caption="Test Image",
                use_container_width=True
            )

        st.write("")

        if st.button("Verify Faces",use_container_width=True):

            with st.spinner("Generating Face Embeddings..."):

                similarity,distance,prediction=verify_faces(
                    image1,
                    image2
                )

            if similarity is None:

                st.error("Face could not be detected in one or both images.")

            else:

                st.markdown("---")

                m1,m2,m3=st.columns(3)

                m1.metric(
                    "Similarity",
                    f"{similarity:.3f}"
                )

                m2.metric(
                    "Euclidean Distance",
                    f"{distance:.3f}"
                )

                m3.metric(
                    "Prediction",
                    prediction
                )

                st.progress(float(similarity))

                st.markdown("### Result")

                if prediction=="Same Person":

                    st.success(
                        "✅ Verification Successful\n\nBoth images belong to the SAME PERSON."
                    )

                    st.balloons()

                else:

                    st.error(
                        "❌ Verification Failed\n\nBoth images belong to DIFFERENT PERSONS."
                    )

                st.markdown("---")

                st.subheader("Interpretation")

                st.write(f"""
Similarity Score : **{similarity:.3f}**

Euclidean Distance : **{distance:.3f}**

Prediction : **{prediction}**
""")
