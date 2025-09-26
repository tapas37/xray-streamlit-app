import streamlit as st
import tensorflow as tf

st.title('Chest X-Ray Classifier')
img_size = 100
MODEL_PATH = "model_vgg_tapas.h5"
CATEGORIES = ["NORMAL", "PNEUMONIA"]

model = tf.keras.models.load_model(MODEL_PATH)

def load_classifier():
    st.subheader("Upload an X-Ray image to detect if it is Normal or Pneumonia")
    file = st.file_uploader("Upload an image", type=['jpg', 'jpeg', 'png'])

    if file is not None:
        img = tf.keras.preprocessing.image.load_img(file, target_size=(img_size, img_size))
        new_array = tf.keras.preprocessing.image.img_to_array(img)
        new_array = new_array.reshape(-1, img_size, img_size, 3)


        st.image(file, caption="Uploaded Image", use_container_width=True)
        if st.button("PREDICT"):
            prediction = model.predict(new_array / 255.0)
            label = CATEGORIES[int(round(prediction[0][0]))]
            confidence = round(float(prediction[0][0]) * 100, 2)
            st.write(f"**Prediction:** {label}")
            st.write(f"**Confidence:** {confidence}%")

if __name__ == "__main__":
    load_classifier()
