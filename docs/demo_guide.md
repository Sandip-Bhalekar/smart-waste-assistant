# Demonstration Guide

1. Create the virtual environment with Python 3.11 and install `requirements.txt`.
2. Confirm `models/best.pt` exists. If it does not, train in Colab using `notebooks/train_waste_model.ipynb` and copy the weights locally.
3. Start the app: `streamlit run app.py`.
4. Show the landing page: mixed-photo upload, no bounding-box language.
5. Upload a photo with at least two materials (for example a bottle and a can).
6. Click **Analyze Waste** once. Wait for the status messages.
7. Point to category cards, counts, confidence, and the checklist. Confirm that no boxes are drawn on the image.
8. Open **Uncertain results** and explain the threshold.
9. Export JSON.
10. Open **History** only if a real analysis was saved. Do not invent statistics.
11. State the limitations: AI predictions can be wrong; municipal rules win; metrics come from the evaluation report, not from the live demo.

If the detector misses a category, say so. Do not hardcode example detections.
