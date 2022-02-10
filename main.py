from torchvision import transforms
import torch
from PIL import Image
import streamlit as st
import math

save_path = './net.pth'


@st.cache(allow_output_mutation=True)
def load_model():
    model = torch.load(save_path, map_location='cpu')
    return model.eval()
with st.spinner('Model is being loaded...'):
    best_model = load_model()


def predict(image):
    transform = transforms.Compose([
      transforms.Resize(256),
      transforms.CenterCrop(224),
      transforms.ToTensor(),
      transforms.Normalize(
        mean=[0.485, 0.456, 0.406],
        std=[0.229, 0.224, 0.225]
       )])

    img = Image.open(image)
    batch_t = torch.unsqueeze(transform(img), 0)

    best_model.eval()
    out = best_model(batch_t)

    classes = ['airport',
               'bridge',
               'building',
               'car',
               'pipe',
               'power plant',
               'railway']

    prob = torch.nn.functional.softmax(out, dim=1)[0] * 100
    _, indices = torch.sort(out, descending=True)
    return [(classes[idx], math.floor(prob[idx].item()))
            for idx in indices[0][:1]]

st.set_option('deprecation.showfileUploaderEncoding', False)
st.title("Image classification App")

file_up = st.file_uploader("Upload an  image", type=["jpg", "jpeg", "png"])

if file_up is not None:
    image = Image.open(file_up)
    st.image(image, caption='Uploaded Image.', use_column_width=True)
    st.write("")
    labels = predict(file_up)

    # print out the top 5 prediction labels with scores
    for i in labels:
        st.write("The model's prediction: ", i[0].upper())
        st.write("Accuracy: ", i[1], "%")
