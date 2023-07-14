# Audio Design Toolkit

Multipage app. 

Run - 
```
cd interface
streamlit run app.py --server.port=<port number>
```

Additional Libraries:
- librosa               0.10.0
- numpy                 < 1.24
- lpips                 0.1.4
- typing-extensions     4.6.3
- pyloudnorm            0.1.1
- mixpanel



https://github.com/streamlit/streamlit/issues/2312#issuecomment-1426169542

sed -i -e 's/),1e3)/),1e4)/g' [PATH_TO_PYTHON_DIR]/lib/python3.8/site-packages/streamlit/static/static/js/main.*.js
sed -i -e 's/baseUriPartsList,500/baseUriPartsList,10000/g' [PATH_TO_PYTHON_DIR]/lib/python3.8/site-packages/streamlit/static/static/js/main.*.js
