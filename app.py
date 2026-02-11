import streamlit as st
import tensorflow as tf
from PIL import Image
import numpy as np
from huggingface_hub import hf_hub_download

# --- 1. Streamlit Page Configuration ---
st.set_page_config(
    page_title="DFU Detection System",
    page_icon="👣",
    layout="wide"
)

# --- 2. Initialize Session State ---
if 'selected_page' not in st.session_state:
    st.session_state.selected_page = "About DFU"

if 'show_dfu_advice' not in st.session_state:
    st.session_state.show_dfu_advice = False

# --- 3. Navigation Functions (Callbacks) ---
def nav_to_detector():
    st.session_state.selected_page = "DFU Detector"

def nav_to_about():
    st.session_state.selected_page = "About DFU"

def nav_to_advice():
    st.session_state.selected_page = "Medical Advice"

# --- 4. Load the Model from Hugging Face ---
@st.cache_resource
def load_my_model():
    try:
        model_path = hf_hub_download(
            repo_id="abdelrahmanemam10/dfu_model",  # الريبو عندك
            filename="dfu_model.keras"              # اسم الملف في الريبو
        )
        return tf.keras.models.load_model(model_path)
    except Exception as e:
        st.error(f"Error loading model: {e}")
        st.stop()

model = load_my_model()
class_names = ['Abnormal (Ulcer)', 'Normal (Healthy skin)']

# --- 5. Sidebar Navigation ---
page_options = ["About DFU", "DFU Detector"]
if st.session_state.show_dfu_advice:
    page_options.append("Medical Advice")

st.sidebar.title("Navigation")
st.sidebar.radio(
    "Go to",
    page_options,
    key="selected_page"
)

# --- Page 1: About DFU ---
if st.session_state.selected_page == "About DFU":
    st.title("About Diabetic Foot Ulcer (DFU) 👣")

    st.markdown("""
    ### What is DFU?
    **Diabetic Foot Ulcer** is a serious complication of diabetes mellitus, typically characterized by a breakdown of skin tissue on the foot.
    
    It is a critical condition that requires immediate attention to prevent chronic infections or even lower-limb amputation.
    """)

    col1, col2 = st.columns(2)

    with col1:
        st.header("Why Does it Happen? 🧬")
        st.write("""
        DFUs are usually caused by a combination of factors:
        * **Peripheral Neuropathy:** Nerve damage that reduces the ability to feel pain, meaning small injuries go unnoticed.
        * **Poor Circulation (Ischemia):** Reduced blood flow to the feet slows down the healing process.
        * **Hyperglycemia:** High blood sugar weakens the immune system's ability to fight bacteria.
        """)

    with col2:
        st.header("Common Symptoms 🤒")
        st.write("""
        * **Skin Discoloration:** Redness, darkening, or bluish tints around an area.
        * **Swelling (Edema):** Unusual inflammation or puffiness in the foot.
        * **Unusual Odors:** A foul smell indicating a potential underlying infection.
        * **Fluid Drainage:** Pus or blood noticed on socks or shoes.
        """)

    st.header("Prevention & Daily Care 🛡️")
    st.info("""
    - **Daily Inspection:** Check the soles of your feet every day using a mirror.
    - **Proper Footwear:** Wear specialized diabetic shoes and avoid walking barefoot.
    - **Sugar Management:** Keeping your HbA1c levels in check is your best defense.
    """)

    st.warning(
        "⚠️ **Disclaimer:** This AI tool is for **preliminary screening and educational purposes only**. It is not a clinical diagnosis. Always consult a medical professional if you suspect an injury."
    )

    st.button("Start DFU Detector", on_click=nav_to_detector)

# --- Page 2: DFU Detector ---
elif st.session_state.selected_page == "DFU Detector":
    st.title("AI-Powered DFU Detector 🔬")
    st.write("Upload a clear photo of the foot area for analysis.")

    uploaded_file = st.file_uploader("Choose an image...", type=["jpg", "jpeg", "png"])

    if uploaded_file is not None:
        # Convert to RGB to handle PNG alpha channels
        image = Image.open(uploaded_file).convert('RGB')
        st.image(image, caption="Uploaded Image", width=400)

        st.write("Analyzing image...")

        img_resized = image.resize((224, 224))
        img_array = tf.keras.preprocessing.image.img_to_array(img_resized)
        img_array = tf.expand_dims(img_array, 0)

        predictions = model.predict(img_array)
        score = tf.nn.softmax(predictions[0])
        predicted_idx = np.argmax(score)
        result = class_names[predicted_idx]
        confidence = 100 * np.max(score)

        if predicted_idx == 0:  # Abnormal Case
            st.markdown(f"<h2 style='color: red;'>Result: {result} 🔴</h2>", unsafe_allow_html=True)
            st.write(f"Confidence Level: {confidence:.2f}%")
            st.session_state.show_dfu_advice = True
            st.warning("⚠️ Abnormality detected. Please proceed to the 'Medical Advice' section.")
            st.button("Go to Medical Advice", on_click=nav_to_advice)
        else:  # Normal Case
            st.markdown(f"<h2 style='color: green;'>Result: {result} ✅</h2>", unsafe_allow_html=True)
            st.write(f"Confidence Level: {confidence:.2f}%")
            st.session_state.show_dfu_advice = False
            st.success("The foot tissue appears normal. Continue with regular care.")

    st.markdown("---")
    st.button("Back to About Page", on_click=nav_to_about)

# --- Page 3: Medical Advice ---
elif st.session_state.selected_page == "Medical Advice":
    st.title("What to do if an Ulcer is Suspected? 🚨")

    st.header("1. Consult a Specialist Immediately 👨‍⚕️")
    st.write("Seek help from a podiatrist or vascular surgeon. Early intervention is vital.")

    st.header("2. Immediate Foot Care 🩹")
    st.markdown("""
    * **Off-loading:** Do not walk or put any pressure on the injured foot.
    * **Sterile Dressing:** Cover the area with a clean, sterile bandage.
    * **No Self-Surgery:** Never attempt to cut calluses or skin around the wound yourself.
    """)

    st.header("3. Monitor Glucose 🩸")
    st.write("Maintain strict blood sugar control to help your body heal faster.")

    st.error("Seek emergency care if you experience spreading redness, high fever, or severe pain.")

    st.button("Return to Detector", on_click=nav_to_detector)
