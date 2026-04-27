# Streamlit Notes

## What is Streamlit?
- **Streamlit** is an open-source framework used to create web apps for Machine Learning and Data Science.
- It can be used to deploy Machine Learning models.
- Using Streamlit, we can create applications with just a few lines of code.

### Run a Streamlit App
```python
streamlit run file_name.py
```

---

# Import Streamlit
```python
import streamlit as st
```

# Streamlit Basic Functions

## 1. `st.title()`
Used to give the title of the page.

```python
st.title("Cold Email")
```

---

## 2. `st.code()`
Used to display code snippets.

```python
st.code("print('Hello')")
```

---

## 3. `st.image()`
Used to display images.

```python
st.image("path")
```

---

## 4. `st.header()`
Used to give a header to the page.

```python
st.header("Machine Learning")
```

---

## 5. `st.subheader()`
Used to give a subheader.

```python
st.subheader("Linear Regression")
```

---

## 6. `st.latex()`
Used to display mathematical equations.

```python
st.latex(r'''a+b x^2+c''')
```

---

## 7. `st.caption()`
Used to display small caption text.

```python
st.caption("hello")
```

---

## 8. `st.info()`
Displays information inside a blue box.

```python
st.info("details")
```

---

## 9. `st.warning()`
Displays warning messages in an orange box.

```python
st.warning("come fast")
```

---

## 10. `st.write()`
Used to display text.

```python
st.write("Akhila Reddy")
```

---

## 11. `st.error()`
Displays error messages in a red box.

```python
st.error("message")
```

---

## 12. `st.success()`
Displays success messages in a green box.

```python
st.success("message")
```

---

## 13. `st.text()`
Displays text in a different font.

```python
st.text("hello")
```

---

## 14. `st.markdown()`
Used for formatting text.

```python
st.markdown("hello")
st.markdown("# hello")     # H1
st.markdown("## hello")    # H2
st.markdown(":fish:")
```

- Increasing `#` decreases font size.
- Can display emojis and formatted text.

---

# Streamlit Widgets (UI Components)

## 1. `st.checkbox()`
Used for checkboxes.

```python
st.checkbox("Login")
```

---

## 2. `st.button()`
Creates a button.

```python
st.button("Click")
```

---

## 3. `st.radio()`
Used to select one option.

```python
st.radio("Pick Gender",["Male","Female"])
```

---

## 4. `st.selectbox()`
Select one item from dropdown.

```python
st.selectbox("Pick Course",["ML","Cloud"])
```

---

## 5. `st.multiselect()`
Select multiple options.

```python
st.multiselect("Choose Dept",["CSE","ECE"])
```

---

## 6. `st.slider()`
Move a pointer in a given range.

```python
st.slider("Enter Number",0,10)
```

---

## 7. `st.select_slider()`
Move pointer among given values.

```python
st.select_slider("Rating",["Bad","Good"])
```

---

## 8. `st.time_input()`
Select time.

```python
st.time_input("Enter Time")
```

---

## 9. `st.number_input()`
Increase/decrease numbers.

```python
st.number_input("Pick Number",0,50)
```

---

## 10. `st.text_input()`
Input box for text.

```python
st.text_input("Enter Name")
```

---

## 11. `st.text_area()`
Large text area.

```python
st.text_area("Enter Address")
```

---

## 12. `st.date_input()`
Calendar date selection.

```python
st.date_input("Date of Birth")
```

---

## 13. `st.file_uploader()`
Upload files.

```python
st.file_uploader("Upload Resume")
```

---

## 14. `st.color_picker()`
Pick colors.

```python
st.color_picker("Color")
```

---

## 15. `st.progress()`
Displays progress bar.

```python
st.progress(90)
```

---

## 16. `st.spinner()`
Shows loading/waiting message.

```python
import time

with st.spinner("Running...Wait"):
    time.sleep(3)
```

---

## 17. `st.balloons()`
Celebration balloons animation.

```python
st.balloons()
```

---

## 18. `st.sidebar`
Creates left sidebar.

```python
st.sidebar.title("Cold Email")
st.sidebar.text_input("Email")
st.sidebar.button("Submit")
```

---

# Data Visualization

```python
import pandas as pd
import numpy as np
```

## Sample Data
```python
data = pd.DataFrame(
np.random.randn(50,2),
columns=["x","y"]
)
```

---

## Bar Chart
Displays data in bar format.

```python
st.bar_chart(data)
```

---

## Line Chart
Displays data as line graph.

```python
st.line_chart(data)
```

---

## Area Chart
Displays data in area graph.

```python
st.area_chart(data)
```

---

# Summary
Streamlit is used for:

- Building web apps  
- Deploying ML models  
- Creating dashboards  
- Data visualization  
- Interactive user interfaces  
