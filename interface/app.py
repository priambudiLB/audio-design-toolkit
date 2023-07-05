from multipage import MultiPage
from perceptual_guided_control import main as pg_control_main
from query_gan import main as query_gan_main
from query_gan_react import main as query_gan_react
from sefa import main as sefa_main
from sefa_react import main as sefa_react
#from GaverAnalysisSynthesis import main as GaverAnalysisSynthesisMain
#from SefaInterface import main as SefaInterfaceMain

# Based on and modified from https://github.com/upraneelnihar/streamlit-multiapps

app = MultiPage()
app.add_app('our-algo', pg_control_main)
app.add_app('algo-1', query_gan_main)
app.add_app('algo-2', sefa_main)
app.add_app('algo1', query_gan_react)
app.add_app('algo2', sefa_react)
app.run()

