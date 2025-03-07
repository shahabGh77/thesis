from cProfile import label
from tracemalloc import stop
from turtle import color
from cv2 import rotate
from matplotlib import pyplot as plt
import numpy as np
from numpy import genfromtxt
import matplotlib as mpl
from matplotlib.patches import ConnectionPatch

# mpl.use("pgf")
# mpl.rcParams.update({
#     'text.latex.preamble': ['\\usepackage{gensymb}'],
#     'image.origin': 'lower',
#     'image.interpolation': 'nearest',
#     'image.cmap': 'gray',
#     'axes.grid': False,
#     'savefig.dpi': 150,  # to adjust notebook inline plot size
#     'axes.labelsize': 8, # fontsize for x and y labels (was 10)
#     'axes.titlesize': 8,
#     'font.size': 8, # was 10
#     'legend.fontsize': 6, # was 10
#     'xtick.labelsize': 8,
#     'ytick.labelsize': 8,
#     'text.usetex': True,
#     'figure.figsize': [3.39, 2.10],
#     'font.family': 'serif',
# })

saveFormat = 'eps'

def set_size(width_pt, fraction=1, subplots=(1, 1)):
    """Set figure dimensions to sit nicely in our document.

    Parameters
    ----------
    width_pt: float
            Document width in points
    fraction: float, optional
            Fraction of the width which you wish the figure to occupy
    subplots: array-like, optional
            The number of rows and columns of subplots.
    Returns
    -------
    fig_dim: tuple
            Dimensions of figure in inches
    """
    # Width of figure (in pts)
    fig_width_pt = width_pt * fraction
    # Convert from pt to inches
    inches_per_pt = 1 / 72.27

    # Golden ratio to set aesthetic figure height
    golden_ratio = (5**.5 - 1) / 2

    # Figure width in inches
    fig_width_in = fig_width_pt * inches_per_pt
    # Figure height in inches
    fig_height_in = fig_width_in * golden_ratio * (subplots[0] / subplots[1])

    return (fig_width_in, fig_height_in)


def iterationLabel(start, end):
    res = []
    for i in range(start, end):
        res.append(f'{i+1}_{i}')
    return res



################## convergence results #################################################
iterations = ['2_1', '3_2', '4_3', '5_4', '6_5', '7_6', '8_7', '9_8', '10_9', '11_10', '12_11', '13_12',
              '14_13', '15_14', '16_15', '17_16', '18_17', '19_18', '20_19', '21_20', '22_21', '23_22',
              '24_23', '25_24', '26_25', '27_26', '28_27', '29_28', '30_29', '31_30', '32_31']
#fraud honest bad good

A40 = np.array([
    [430383.3970863391, -430369.3974032663, 6275555.936357554, -6275555.936205398]
    ,[1202.0630564820021, 162.9370287321508, 62361.43212624267, -62220.43210672587]
    ,[-92.53690548427403, -143.46309332177043, 1143.2443061359227, -141.2443060912192]
    ,[-256.9423933364451, 256.9423927925527, 80.54487933963537, -80.54487874731421]
    ,[-6.4965638387948275, 6.496563404798508, -22.185355007648468, 22.18535467982292]
    ,[365.6665799114853, -365.6665794476867, 12.945955868810415, -12.945955734699965]
    ,[-346.8390590045601, 346.83905859291553, -18.863652288913727, 18.86365206167102]
    ,[-45.801096219569445, 48.80109590664506, 33.13974619284272, -33.139745846390724]
    ,[27.980379400774837, -23.980379339307547, -41.26153698191047, 48.26153660938144]
    ,[382.8735039476305, -382.87350419163704, 32.96065851673484, -32.96065894514322]
    ,[-373.3921709358692, 373.3921712860465, -13.81527240946889, 13.81527329608798]
    ,[-45.05675878375769, 45.056759137660265, 29.749175917357206, -29.749176163226366]
    ,[27.960824800655246, -27.960825342684984, -41.61771873384714, 41.61771863698959]
    ,[-35.77882851473987, 35.77882904559374, 32.98090195283294, -32.98090187460184]
    ,[35.78028035722673, -35.78028080612421, -54.02813268825412, 54.02813268825412]
    ,[-37.7124353852123, 37.71243616938591, 53.91981002315879, -53.91981031000614]
    ,[37.55806018412113, -37.558060240000486, -38.39437213540077, 38.39437199383974]
    ,[-30.064005877822638, 30.064005568623543, 37.850461818277836, -37.850461419671774]
    ,[663.0958205331117, 477.90417870879173, -34.59809383004904, 34.598093677312136]
    ,[-149.16600083187222, -94.83399895206094, 490.0561427362263, 406.9438576884568]
    ,[0.37495618872344494, -0.37495628744363785, -37.46762469410896, 37.46762428805232]
    ,[246.06071811541915, -246.06071877107024, 20.45692888647318, -20.456928923726082]
    ,[66.55491265840828, -66.5549121312797, 8.32363609969616, -8.323635689914227]
    ,[-273.1769580580294, 273.1769589483738, 55.923246923834085, -55.92324773967266]
    ,[-9.19512335024774, 9.195123374462128, -63.582560036331415, 63.58256030455232]
    ,[-32.05629969201982, 32.056299556046724, 9.9430420845747, -9.943042077124119]
    ,[28.275401383638382, -28.27540212124586, -33.261288940906525, 33.261289454996586]
    ,[219.56265304237604, -219.56265268847346, 35.24992699176073, -35.249927785247564]
    ,[-180.17961224913597, 180.17961278185248, -4.4393215253949165, 4.439321704208851]
    ,[-63.62547932192683, 63.62547941878438, 28.402297794818878, -28.402297511696815]
    ,[25.209750413894653, -25.209750957787037, -59.504643976688385, 59.5046437010169]
    ,[-363.25335974805057, 363.2533598728478, 34.84008343890309, -34.84008331224322]
    ,[300.8265806231648, -300.8265804424882, -73.44763420149684, 73.44763404130936]
    ,[32.161999978125095, -32.16199945285916, 56.660834819078445, -56.66083500161767]
    ,[30.69638136588037, -30.696382470428944, -18.67462619766593, 18.674626667052507]
    ,[-28.62624337337911, 28.626243963837624, 36.07439478114247, -36.07439521700144]
    ,[274.83447042666376, -274.8344697020948, -34.264106180518866, 34.26410623639822]
    ,[-236.45428705960512, 236.45428638905287, 63.939631670713425, -63.93963162600994]
    ,[-7.277765056118369, 7.277765356004238, -40.33832674100995, 40.33832684904337]
    ,[-31.105884620919824, 31.105883542448282, 9.389692824333906, -9.389692775905132]
    ,[27.167852211743593, -27.167851503938437, -34.0031982883811, 34.003198217600584]
    ,[-27.19878989830613, 27.198789931833744, 34.97697978839278, -34.97697975113988]
    ,[18.1899043507874, -18.18990458920598, -34.06709196045995, 34.06709188967943]
    ,[-18.972397930920124, 18.972397323697805, 31.51517452299595, -31.515173990279436]
    ,[28.777527233585715, -28.777526676654816, -31.819744125008583, 31.819743644446135]
    ,[-30.222040684893727, 30.222040604799986, 34.69532985240221, -34.69532973691821]
    ,[30.429853348061442, -30.429852955043316, -36.887928403913975, 36.88792822882533]
    ,[-28.161470329388976, 28.16146981716156, 36.98567432910204, -36.98567433655262]
    ,[28.15678326971829, -28.156783260405064, -34.53100210800767, 34.531002797186375]
    ,[-28.159599494189024, 28.159599971026182, 34.51351926475763, -34.51352024078369]
    ,[27.453815983608365, -27.45381562039256, -34.52770618349314, 34.52770667523146]
    ,[-29.654536264017224, 29.65453627705574, 34.21385335177183, -34.21385354548693]
    ,[30.360376013442874, -30.360376831144094, -36.60188481211662, 36.60188554599881]
    ,[387.5854442138225, -387.5854444094002, 36.915049996227026, -36.915050614625216]
    ,[-371.3496859073639, 371.3496854752302, -16.548381078988314, 16.548381306231022]
    ,[91.73773335106671, -91.73773279413581, 26.50668926909566, -26.50668940320611]
    ,[-103.28930519893765, 103.28930600360036, -100.94729658961296, 100.94729619845748]
    ,[-32.54717637784779, 32.54717490449548, 94.26918259635568, -94.26918188855052]
    ,[27.92413435317576, -27.924133766442537, -37.36105041950941, 37.36105014756322]
    ,[-38.762445697560906, 38.762445852160454, 34.082942731678486, -34.082942839711905]
    ,[38.66979272477329, -38.669792018830776, -39.34752621129155, 39.34752590581775]
    ,[-28.138954985886812, 28.138954304158688, 39.20886826887727, -39.20886805653572]
    ,[22.442246479913592, -22.442246053367853, -34.45382723212242, 34.453827660530806]
    ,[224.3415055014193, -224.3415050022304, 26.977269265800714, -26.97726983577013]
    ,[-181.24207517132163, 181.2420746050775, 2.626289751380682, -2.6262892112135887]
    ,[-62.64869797043502, 62.64869746938348, 28.32189577445388, -28.32189602404833]
    ,[271.5288140308112, -271.5288142412901, -59.342969954013824, 59.34297015145421]
    ,[-237.14465405791998, 237.14465473219752, 64.461131490767, -64.46113239601254]
    ,[-13.667725328356028, 13.667724546045065, -39.82495082169771, 39.82495187968016]
    ,[79.08447417616844, -79.0844729617238, 7.684213798493147, -7.684214159846306]
    ,[-45.23053218796849, 45.23053149133921, -67.20077792555094, 67.20077829807997]
    ,[-54.989271899685264, 54.98927165567875, 90.99995017796755, -90.99995071440935]
    ,[25.09628040716052, -25.096279330551624, -56.36923538148403, 56.369235549122095]
    ,[-28.531670847907662, 28.53166950494051, 34.637524243444204, -34.637523885816336]
    ,[29.072793191298842, -29.07279333844781, -33.336311895400286, 33.336311623454094]
    ,[-28.08266556635499, 28.08266657218337, 34.932971969246864, -34.93297182023525]
    ,[443.8855322152376, -443.8855323381722, -34.51669716089964, 34.5166969075799]
    ,[549.1032254956663, 1451.8967743963003, 52.50544184073806, -52.50544197112322]
    ,[-89.93713411130011, -305.06286614760756, 639.7607705704868, 966.2392295375466]
    ,[-28.12795388698578, 28.12795363366604, 104.7150208465755, -104.71502090990543]
    ,[28.242566108703613, -28.242565482854843, -34.90539437532425, 34.90539413318038]
    ,[-28.465460726991296, 28.465460069477558, 35.23068152740598, -35.230681233108044]
    ,[28.134606048464775, -28.134605813771486, -34.463022731244564, 34.46302257850766]
    ,[-27.825068280100822, 27.825068090111017, 34.57658338919282, -34.57658336311579]
    ,[28.166100347414613, -28.1660999879241, -34.439201194792986, 34.4392009600997]
    ,[-28.155354036018252, 28.15535420551896, 34.5372098274529, -34.53720968961716]
    ,[28.156989689916372, -28.156990319490433, -34.53396559134126, 34.533965557813644]
    ,[-28.15749929845333, 28.1574995405972, 34.535358157008886, -34.53535808250308]
    ,[399.11885094456375, -399.1188508719206, -34.53570174053311, 34.535701774060726]
    ,[-384.044140227139, 384.0441406480968, 39.43100382760167, -39.43100392445922]
    ,[13.803334875032306, -13.803334832191467, -30.09083602577448, 30.090836081653833]
    ,[-28.752411423251033, 28.752410769462585, 26.089053984731436, -26.089054241776466]
    ,[28.043057112023234, -28.043056704103947, -35.39133397862315, 35.39133444055915]
    ,[-28.168347273021936, 28.168347097933292, 34.5099687166512, -34.50996905565262]
    ,[28.156911805272102, -28.156912218779325, -34.55007753521204, 34.55007755756378]
    ,[-28.15767591819167, 28.157676424831152, 34.53763008490205, -34.537630163133144]
    ,[27.82912066951394, -27.82912114262581, -34.535740185528994, 34.535740591585636]
    ,[-29.165931567549706, 29.16593225300312, 34.20617501437664, -34.206175584346056]
    ,[29.45643288269639, -29.456434056162834, -34.893197041004896, 34.893197413533926]
    ,[-35.660833302885294, 35.6608339138329, 35.210539035499096, -35.210539188236]
    ,[35.81308008171618, -35.81308006495237, -53.81410963088274, 53.814109958708286]
    ,[-28.251699725165963, 28.25170037522912, 53.97153814136982, -53.97153853625059]
    ,[-305.59181938134134, 305.59181923791766, -34.63202637806535, 34.632027104496956]
    ,[270.20402183011174, -270.20402255281806, -4.53477930650115, 4.534778688102961]
    ,[61.455218294635415, -61.455216865986586, 0.05132262781262398, -0.05132301524281502]
    ,[390.12328704819083, -390.12328815460205, 37.39470862597227, -37.39470819756389]
    ,[-371.86293476447463, 371.8629351556301, -15.085902098566294, 15.085901718586683]
    ,[-43.660208847373724, 43.660208858549595, 26.616117540746927, -26.616117171943188]
    ,[299.01309866830707, -299.0130981877446, -43.58241803944111, 43.58241807296872]
    ,[-288.23699831590056, 288.23699824884534, 6.3462589755654335, -6.346259102225304]
    ,[17.27529109083116, -17.275290858000517, 0.08259059116244316, -0.08259031176567078]
    ,[-28.633820282295346, 28.633820049464703, 27.94845524057746, -27.9484554938972]
    ,[28.061791563406587, -28.06179242953658, -35.25410905107856, 35.25410948321223]
    ,[387.5911554247141, -387.59115474671125, 34.51506595313549, -34.51506621018052]
    ,[-371.35146572999656, 371.3514651246369, -16.558008830994368, 16.55800899490714]
    ,[-43.64176204241812, 43.641761630773544, 26.51951965689659, -26.519519809633493]
    ,[22.711874645203352, -22.711873918771744, -43.5708434432745, 43.57084375992417]
    ,[-23.483198665082455, 23.48319823294878, 31.506984867155552, -31.506985317915678]
    ,[28.162481658160686, -28.16248118132353, -32.485528100281954, 32.48552840948105]
])

C20 = np.array([
    [176979.7266862318, -176979.72671467625, 2794319.623128539, -2794319.6230830867],
    [246.5629827650264, -246.56298644654453, 26552.293230511248, -26552.293233240023],
    [42.611956109292805, -42.61195634678006, 445.7059060037136, -445.70590581558645],
    [-138.20587848778814, 138.2058790512383, 46.96608855202794, -46.96608906425536],
    [146.40222577378154, -146.40222601220012, -42.240906378254294, 42.24090678058565],
    [-0.8035379750654101, 0.8035381138324738, 52.230570174753666, -52.23057038150728],
    [2.1733598802238703, -2.17336024902761, 2.558715097606182, -2.558715185150504],
    [-0.7628892669454217, 0.7628893386572599, 1.7515717595815659, -1.7515713926404715],
    [1.4311093967407942, -1.4311090894043446, -0.5992301851511002, 0.5992301292717457],
    [-0.15768015757203102, 0.15768012590706348, 0.008212210610508919, -0.008212128654122353],
    [-0.2913944572210312, 0.29139461927115917, 0.19444148987531662, -0.19444161653518677],
    [-0.08514275215566158, 0.08514260686933994, 0.2218083143234253, -0.22180833853781223],
    [-0.28567831963300705, 0.2856782879680395, -0.03282305970788002, 0.032822923734784126],
    [0.49384436942636967, -0.4938444159924984, -0.24205979891121387, 0.24205969460308552],
    [-0.05668589472770691, 0.05668606236577034, 0.11438808403909206, -0.11438768543303013],
    [-0.06162427365779877, 0.06162412278354168, 0.056286074221134186, -0.05628632381558418],
    [-5.471056064590812, 5.471056010574102, 0.024156004190444946, -0.0241559948772192],
    [5.440081254579127, -5.440081411972642, -1.7431644853204489, 0.7431643679738045], # -1
    [0.06561763677746058, -0.06561729870736599, 1.668494340032339, -0.6684941407293081], # +1
])
C10 = np.array([
    [94648.84266906558, -94648.84272870142, 866366.1052641897, -866366.1052608686],
    [417.3010920267552, -417.30109205376357, 24419.40149138961, -24419.401485950686],
    [-60.9230593633838, 60.92305989097804, 399.44802972022444, -399.44802962150425],
    [15.67197442194447, -15.671974775381386, 30.540703269653022, -30.540703244507313],
    [-32.15396521333605, 32.15396507550031, -4.6536539401859045, 4.653653905726969],
    [33.28265312965959, -33.28265294525772, 9.627350831404328, -9.62735085375607],
    [-34.070010049734265, 34.07000992074609, -2.8920359602198005, 2.892035939730704],
    [33.898292587604374, -33.89829276688397, 8.829013123176992, -8.829013072885573],
    [-35.30814085807651, 35.30814082548022, -6.868205493316054, 6.868205562233925],
    [35.45793459424749, -35.4579346422106, 7.366024355404079, -7.366024397313595],
    [-34.52067752368748, 34.520677451975644, -7.437441556714475, 7.437441503629088],
    [34.64333126042038, -34.64333101827651, 7.902098972350359, -7.902098995633423],
    [-35.306146376766264, 35.30614620260894, -7.7173460545018315, 7.717346087098122],
    [35.42119849100709, -35.42119827773422, 7.600778506137431, -7.600778479129076],
    [-34.858383709099144, 34.858383456245065, -7.670270001515746, 7.67026998847723],
    [34.7877612747252, -34.78776132594794, 7.779130714945495, -7.779130761511624],
    [-35.22639860212803, 35.22639869339764, -7.719678261317313, 7.719678303226829],
    [35.32692628679797, -35.32692620437592, 7.615305375307798, -7.615305385552347],
    [-34.9628108041361, 34.96281066443771, -7.641251896508038, 7.6412518080323935],
])
C8 = np.array([
    [169527.64837652864, -169527.64831832144, 1117123.0111336675, -1117123.011137749],
    [3704.9935822286643, -3704.993580537848, 39287.97612658609, -39287.97612258326],
    [100.81037314562127, -100.81037333700806, 1055.583864000626, -1055.583863824606],
    [15.201177256647497, -15.20117714535445, 72.9056414840743, -72.90564158651978],
    [-6.128753179684281, 6.128753304481506, 18.935589188709855, -18.93558911420405],
    [-8.173284079879522, 8.173284004442394, 6.268144789617509, -6.268144714646041],
    [-4.2085133283399045, 4.208513181656599, 3.746524526271969, -3.7465245574712753],
    [-4.613050248473883, 4.613050512038171, 1.7411396214738488, -1.7411397453397512],
    [-1.712693028151989, 1.7126930439844728, 0.4797729062847793, -0.4797728806734085],
    [-1.1727504110895097, 1.1727502578869462, 0.4523472934961319, -0.4523472525179386],
    [-0.28893460985273123, 0.28893457632511854, -0.029493490234017372, 0.029493541456758976],
    [-0.42126484541222453, 0.4212649120017886, 0.16198451491072774, -0.16198460198938847],
    [-0.0538206179626286, 0.05382056999951601, -0.06750231143087149, 0.06750231236219406],
    [-0.26111875334754586, 0.2611188283190131, 0.08654889138415456, -0.086548768915236],
    [-0.01347441179677844, 0.01347438246011734, -0.05151280527934432, 0.05151271726936102],
    [-0.16688429471105337, 0.16688434779644012, 0.058178188279271126, -0.058178285136818886],
    [-0.0007771290838718414, 0.000777101144194603, -0.038582722656428814, 0.038582781329751015],
    [-0.10794023983180523, 0.10794027429074049, 0.03567667258903384, -0.03567661810666323],
    [0.0008908221498131752, -0.0008908407762646675, -0.02949409279972315, 0.02949402667582035],
])
C2_5 = np.array([
    [21250.303952377406, -21250.303961907513, 253389.84185957303, -253389.8418589104],
    [-213.398547654273, 213.39854755508713, 4544.6901629816275, -4544.690162688028],
    [-12.754772866377607, 12.754772930406034, 64.22609135648236, -64.22609133459628],
    [-90.46308697899804, 90.46308699273504, 5.521706298692152, -5.521706302650273],
    [15.508183546364307, -15.508183509577066, 3.2151675110217184, -3.2151675103232265],
    [-43.904264603508636, 43.904264523647726, 2.7596027192194015, -2.759602735284716],
    [16.57364072301425, -16.573640678077936, -0.31411814177408814, 0.31411814922466874],
    [-22.599860337213613, 22.599860320566222, 0.21172257605940104, -0.2117225769907236],
    [7.811478419462219, -7.811478398973122, 1.8820005634333938, -1.8820005501620471],
    [-8.765527850482613, 8.76552784559317, 0.4290598910301924, -0.42905989987775683],
    [2.751998794148676, -2.7519988021813333, 1.662835568189621, -1.6628355784341693],
    [-2.790388364577666, 2.7903883766848594, 0.17843287298455834, -0.17843286180868745],
    [0.8476044815033674, -0.8476044547278434, 0.7320780504960567, -0.7320780465379357],
    [-0.8417200808180496, 0.841720073018223, 0.08461304754018784, -0.08461306104436517],
    [0.26065727323293686, -0.260657309088856, 0.2450161105953157, -0.245016114320606],
    [-0.24804522935301065, 0.24804526683874428, 0.02964470535516739, -0.02964469650760293],
    [0.07040057657286525, -0.07040056586265564, 0.07516495371237397, -0.075164960231632],
    [-0.07730041013564914, 0.07730039278976619, 0.006745856953784823, -0.00674583250656724],
    [0.019140403368510306, -0.019140413496643305, 0.023260635090991855, -0.02326065395027399],
])

def degDist():
    fig, ax = plt.subplots()
    user = genfromtxt('data/figures/videoGames_users_deg.csv', dtype=int, delimiter=',', skip_header=1)
    product = genfromtxt('data/figures/videoGames_products_deg.csv', dtype=int, delimiter=',', skip_header=1)

    print(product[:, 0])
    plt.xlabel('Degrees')
    plt.ylabel('Frequency')
    plt.yscale('log')
    plt.xscale('log')
    ax.scatter(user[:, 0], user[:, 1], label="Users")
    ax.scatter(product[:, 0], product[:, 1], label="Products")

    # ax.plot(activity, cat, label="cat")
    ax.legend()
    # plt.savefig('reports/paper/pics/degDist.pgf')
    plt.savefig(f'reports/paper/pics/degDist.{saveFormat}')
    plt.show()


def runningTimes():
    fig, ax = plt.subplots()
    iterations = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10]
    M40 = [199, 201, 191, 186, 210, 196, 197, 189, 205, 178]
    M20 = [80, 76, 75, 76, 77, 75, 74, 74, 78, 74]
    M10 = [51, 57, 58, 59, 57, 58, 58, 56, 60, 57]
    M8 = [39, 35, 35, 37, 37, 34, 37, 36, 38, 36]
    M2_5 = [19, 15, 14, 14, 14, 14, 13, 14, 13, 13]

    plt.xlabel('iteration number')
    plt.ylabel('time')
    # ax.plot(iterations, M40, label="40M")
    ax.plot(iterations, M20, label="20M")
    ax.plot(iterations, M10, label="10M")
    ax.plot(iterations, M8, label="8M")
    ax.plot(iterations, M2_5, label="2.5M")

    ax.legend()
    plt.savefig(f'reports/paper/pics/runningTimes.{saveFormat}')
    plt.show()
    # plt.savefig('runningTimes.pgf')
    

def runningTime2():
    fig, ax = plt.subplots()
    numberOfEdges = [0.1, 2.5, 8, 10, 20, 40]

    rTime = [0.493, 14.3, 36.4, 57.1, 75.9, 195.2]
    plt.xlabel('Number of edges(Million)')
    plt.ylabel('Time per iteration(sec)')

    ax.plot(numberOfEdges, rTime, label='Our work')
    ax.plot(np.array(numberOfEdges), np.array(numberOfEdges)*100, label='Akoglu et al.')
    plt.plot(1.15, 120, 'C1o')
    plt.plot(40, 4000, 'C1o')
    plt.plot(20, 75.9, 'C0o')
    plt.plot(40, 195.2, 'C0o')

    ax.annotate(
        '1.15M edges,\nin 120s',
        xy=(1.15, 120), xycoords='data',
        xytext=(+30, 90), textcoords='offset points',
        bbox=dict(boxstyle="round", fc="0.8"),
        arrowprops=dict(arrowstyle="->",
                        connectionstyle="angle,angleA=0,angleB=90,rad=10"))

    ax.annotate(
        '20M edges,\nin 75.9s',
        xy=(20, 75.9), xycoords='data',
        xytext=(+10, 30), textcoords='offset points',
        bbox=dict(boxstyle="round", fc="0.8"),
        arrowprops=dict(arrowstyle="->",
                        connectionstyle="angle,angleA=0,angleB=90,rad=10"))

    ax.annotate(
        '40M edges,\nin 195.2s',
        xy=(40, 195.2), xycoords='data',
        xytext=(-80, 120), textcoords='offset points',
        bbox=dict(boxstyle="round", fc="0.8"),
        arrowprops=dict(arrowstyle="->",
                        connectionstyle="angle,angleA=0,angleB=90,rad=10"))

    ax.annotate(
        '40M edges,\nin 4000s',
        xy=(40, 4000), xycoords='data',
        xytext=(-130, -40), textcoords='offset points',
        bbox=dict(boxstyle="round", fc="0.8"),
        arrowprops=dict(arrowstyle="->",
                        connectionstyle="angle,angleA=90,angleB=0,rad=10"))
    # ax.text(1.15, 120, 120, size=12)
    ax.legend()
    # plt.savefig('reports/paper/pics/runningTimes2.pgf')
    plt.savefig(f'reports/paper/pics/runningTimes2.{saveFormat}')
    plt.show()


def convergenceOne():
    #for electronics 20M
    fig, ax = plt.subplots()

    plt.xlabel('iteration i, i+1', labelpad=-48)
    plt.ylabel('change')
    plt.yscale('symlog')

    ax.plot(iterationLabel(1,120), A40[:, 0], label="userToProduct-messages")
    # ax.plot(iterationLabel(1, 120), A40[:, 1], label="honest-messages")
    # ax.plot(iterationLabel(1, 120), A40[:, 2], label="bad-messages")
    # ax.plot(iterationLabel(1, 120), A40[:, 3], label="good-messages")
    ax.axhline(y=0, linestyle='dotted', color='k', label='convergence')

    plt.xticks(ax.get_xticks()[::5], rotation=80, fontsize='x-small')
    ax.legend()
    # plt.savefig('reports/paper/pics/convergenceOne.pgf')
    plt.savefig(f'reports/paper/pics/convergenceOne.{saveFormat}')
    plt.show()




def convergenceAll():
    fig, ax = plt.subplots()
    

    plt.xlabel('iteration i, i+1')
    ax.xaxis.set_label_coords(0.2, 0.06)
    plt.ylabel('change of productToUser messages')
    plt.yscale('symlog')

    ax.plot(iterationLabel(1, 20), C20[:, 3], label="Electronics")
    ax.plot(iterationLabel(1, 20), C10[:, 3], label="CellPhonesAndAccessories")
    ax.plot(iterationLabel(1, 20), C8[:, 3], label="ToysAndGame")
    ax.plot(iterationLabel(1, 20), C2_5[:, 3], label="VideoGames")
    ax.axhline(y=0, linestyle='dotted', color='k', label='convergence')
    ax.legend()
    plt.xticks(rotation = 80)
    # plt.savefig('reports/paper/pics/convergenceAll.pgf')
    plt.savefig(f'reports/paper/pics/convergenceAll.{saveFormat}')
    plt.show()



def deviationRange():
    fig, ax = plt.subplots()
    # iterations = ['2_1', '3_2', '4_3', '5_4', '6_5', '7_6', '8_7', '9_8', '10_9']
    iterations = ['2_1', '3_2', '4_3', '5_4', '6_5', '7_6', '8_7', '9_8', '10_9',
                  '11_10', '12_11', '13_12', '14_13', '15_14', '16_15', '17_16',
                  '18_17', '19_18', '20_19', '21_20', '22_21', '23_22']
    # ax2 = ax.twinx()
    # plt.style.use('fivethirtyeight')
    plt.xlabel('range of difference between sum(productToUser_messages) \n in iteration i+1 and i ')
    plt.ylabel('range of difference between sum(userToProduct messages) \n in iteration i+1 and i ')


    # plt.xlabel('Honest', loc='left')
    # plt.xlabel('fraud', loc='right')

    plt.yscale('symlog')
    plt.xscale('symlog')
    # ax.spines['left'].set_position('center')
    # ax.spines['bottom'].set_position('center')
    # ax.spines['right'].set_color('none')
    # ax.spines['top'].set_color('none')

    # ax.scatter(['fraud', 'honest', 'bad', 'good'], C20[0], label="20M-good-messages")
    # ax.grid(True)
    # ax.scatter(C20[:, 0], 9*[0], label="20M-fraud-messages")
    # ax.scatter(C20[:, 1], 9*[0], label="20M-honest-messages")

    # ax.scatter(9*[0], C20[:, 2], label="20M-badd-messages")
    # ax.scatter(9*[0], C20[:, 3], label="20M-good-messages")

    # [176979.7266862318, -176979.72671467625, 2794319.623128539, -2794319.6230830867]
    # print(len(A40))
    for i, itr in enumerate(C20[:10]):
        print(i)
        ax.plot([itr[0], 0, itr[1], 0, itr[0]], [0, itr[2], 0, itr[3], 0], label=iterations[i])
        # break
    # for i, txt in enumerate(iterations):
    #     ax.annotate(txt, (C20[:, 0][i], C20[:, 1][i]))
    ax.legend()
    # plt.savefig('deviationRange.pgf')
    plt.tight_layout()
    plt.savefig(f'reports/paper/pics/deviationRange.{saveFormat}')
    plt.show()

def circleConvergenge():
    fig, ax = plt.subplots()
    # plt.scatter( 0 , 0 , s = 7000 )
    # plt.title( 'Circle' )
    # iterations = ['2_1', '3_2', '4_3', '5_4', '6_5', '7_6', '8_7', '9_8', '10_9']
    
    # plt.xlim( -0.85 , 0.85 )
    # plt.ylim( -0.95 , 0.95 )
    plt.yscale('symlog')
    plt.xscale('symlog')
    
    theta = np.linspace( 0 , 2 * np.pi , 150 )
 
    radius = 0.4
    
    a = radius * np.cos( theta )
    b = radius * np.sin( theta )
    
    for i, r in enumerate(C20[:, 0]):
        radius = abs(r)
        ax.plot( radius * np.cos( theta ), radius * np.sin( theta ) , label=iterations[i])
    ax.set_aspect( 1 )
    ax.legend()
    # plt.savefig('circleConvergenge.pgf')
    plt.savefig(f'reports/paper/pics/circleConvergenge.{saveFormat}')
    plt.show()


    # plt.show()


def fraudDist():
    fig, ax = plt.subplots()
    ax.bar(np.linspace(0,1,11)-.02, [1055156, 133115, 80165, 94441, 21168, 20125, 210746, 16851, 26218, 53934, 98247], .05) #c
    ax.bar(np.linspace(0,1,11)-.01, [264191, 18537, 6901, 16064, 2871, 1914, 39046, 1816, 3870, 7024, 18328], .04) #v
    ax.bar(np.linspace(0,1,11)+.01, [2357213, 193199, 75287, 164737, 32869, 17359, 344650, 18186, 45389, 78299, 183248], .03) #e
    ax.bar(np.linspace(0,1,11)+.02, [935216, 92417, 57486, 49006, 10789, 13242, 96786, 8530, 11620, 23065, 41079], .02) #t
    plt.show()


def catgDist():
    fig, ax = plt.subplots()

    size = 0.3
    #                   fraud, honest         good, bad
    vals = np.array([[3740186, 12725302], [1781657, 261080]])

    cmap = plt.get_cmap('tab20c')
    outer_colors = cmap(np.arange(3)*4)
    grp_names = ['Users\n 89%', 'Products\n    11%']
    subgrp_names = ['Fruadster', 'Honest', "Good", 'Bad']
    inner_colors = cmap([1, 2, 5, 6, 9, 10])
    
    wedges, texts = ax.pie(vals.flatten(), radius=1, colors=inner_colors, labels=['22%', '78%', '87%', '13%'], labeldistance=0.8,
        wedgeprops=dict(width=size, edgecolor='w'))

    bbox_props = dict(boxstyle="square,pad=0.3", fc="w", ec="k", lw=0.72)
    kw = dict(arrowprops=dict(arrowstyle="-"),
            bbox=bbox_props, zorder=0, va="center")

    for i, p in enumerate(wedges):
        ang = (p.theta2 - p.theta1)/2. + p.theta1
        y = np.sin(np.deg2rad(ang))
        x = np.cos(np.deg2rad(ang))
        horizontalalignment = {-1: "right", 1: "left"}[int(np.sign(x))]
        connectionstyle = "angle,angleA=0,angleB={}".format(ang)
        kw["arrowprops"].update({"connectionstyle": connectionstyle})
        ax.annotate(subgrp_names[i], xy=(x, y), xytext=(1.35*np.sign(x), 1.4*y),
                    horizontalalignment=horizontalalignment, **kw)

    ax.pie(vals.sum(axis=1), radius=1-size, colors=outer_colors, labels=grp_names, labeldistance=0.615, textprops={'fontsize': 8.5, 'color': 'w'},
        wedgeprops=dict(width=size, edgecolor='w'))

    # plt.legend(loc=(0.9, 0.1))
    # handles, labels = ax.get_legend_handles_labels()
    # subgroup_names_legs = ['Users:Fruadster', 'Users:Honest', 'Products:Bad', 'Products:Good']
    # ax.legend(handles, subgroup_names_legs, loc=(0.9, 0.1))
    plt.text(0, 0, "ٔNodes", ha='center', va='center', fontsize=12)
    plt.tight_layout()
    # plt.savefig('reports/paper/pics/nodesDist.pgf')
    plt.savefig(f'reports/paper/pics/nodesDist.{saveFormat}')
    plt.show()


def survey(results, category_names):
        """
        Parameters
        ----------
        results : dict
            A mapping from question labels to a list of answers per category.
            It is assumed all lists contain the same number of entries and that
            it matches the length of *category_names*.
        category_names : list of str
            The category labels.
        """
        labels = list(results.keys())
        data = np.array(list(results.values()))
        data_cum = data.cumsum(axis=1)
        cmap = plt.get_cmap('RdYlGn')
        category_colors = cmap(
            np.linspace(0.15, 0.85, data.shape[1]))

        fig, ax = plt.subplots(figsize=(10, 5))
        ax.invert_yaxis()
        ax.xaxis.set_visible(False)
        ax.set_xlim(0, np.sum(data, axis=1).max())
        

        for i, (colname, color) in enumerate(zip(category_names, category_colors)):
            widths = data[:, i]
            starts = data_cum[:, i] - widths
            rects = ax.barh(labels, widths, left=starts, height=0.5,
                            label=colname, color=color)

            r, g, b, _ = color
            text_color = 'white' if r * g * b < 0.5 else 'darkgrey'
            ax.bar_label(rects, label_type='center', color=text_color)
        ax.legend(ncol=len(category_names), bbox_to_anchor=(0, 1),
                loc='lower left', fontsize='small')

        return fig, ax

def _plotReviewDist(results, cmap, labels, saveLoc):
    category_names = ['Fake', 'Uncertain', 'Genuine']
    fig, ax = survey(results, category_names)

    cbar = fig.colorbar(mpl.cm.ScalarMappable(cmap=cmap), ticks=[1, 0], pad=0.18, location = 'left', ax=ax)
    cbar.ax.set_yticklabels(labels, fontsize='small')


    plt.tight_layout()
    plt.savefig(saveLoc)
    plt.show()

def reviewDist():
    #honest users
    results = {'0.9 ≤ honest ≤ 1.0': [0.0, 1.0, 99.0], '0.8 ≤ honest < 0.9': [0.0, 9.0, 91.0],
     '0.7 ≤ honest < 0.8': [22.0, 17.0, 61.0], '0.6 ≤ honest < 0.7': [15.0, 35.0, 50.0], 
     '0.5 ≤ honest < 0.6': [24.0, 27.0, 49.0], '0.4 ≤ honest < 0.5': [38.0, 21.0, 41.0], 
     '0.3 ≤ honest < 0.4': [49.0, 5.0, 46.0], '0.2 ≤ honest < 0.3': [53.0, 16.0, 31.0], 
     '0.1 ≤ honest < 0.2': [89.0, 11.0, 0.0], '0.0 ≤ honest < 0.1': [94.0, 1.0, 5.0]}
    labels = ['The most \nhonest user', 'The least \nhonest user']
    cmap = 'Greens'
    # saveLoc = 'reports/paper/pics/honestReviewFreq.pgf'
    saveLoc = f'reports/paper/pics/honestReviewFreq.{saveFormat}'

    # saveLoc = 'honestReviewFreq.png'
    _plotReviewDist(results, cmap, labels, saveLoc)

    #bad products
    results = {'0.9 ≤ bad ≤ 1.0': [16.0, 16.0, 68.0], '0.8 ≤ bad < 0.9': [10.0, 27.0, 63.0],
    '0.7 ≤ bad < 0.8': [18.0, 54.0, 28.0], '0.6 ≤ bad < 0.7': [24.0, 33.0, 43.0],
    '0.5 ≤ bad < 0.6': [9.0, 73.0, 18.0], '0.4 ≤ bad < 0.5': [17.0, 39.0, 44.0],
    '0.3 ≤ bad < 0.4': [20.0, 33.0, 47.0], '0.2 ≤ bad < 0.3': [11.0, 59.0, 30.0], 
    '0.1 ≤ bad < 0.2': [4.0, 24.0, 72.0], '0.0 ≤ bad < 0.1': [13.0, 2.0, 85.0]}
    labels = ['The Worst product', 'The best product']
    cmap = 'Reds'
    # saveLoc = 'reports/paper/pics/badReviewFreq.pgf'
    saveLoc = f'reports/paper/pics/badReviewFreq.{saveFormat}'

    # saveLoc = 'badReviewFreq.png'
    _plotReviewDist(results, cmap, labels, saveLoc)

def reviewDist2():
    # Pie chart, where the slices will be ordered and plotted counter-clockwise:
    # labels = 'Fake', 'Uncertain', 'Genuine' 
    # sizes = [13, 4, 83]
    cmap = plt.get_cmap('RdYlGn')
    colour = cmap(np.linspace(0.11, 0.89, 3))

    # fig1, ax1 = plt.subplots()
    # ax1.pie(sizes, labels=labels, autopct='%1.1f%%', startangle=90, colors=colour)
    # ax1.axis('equal')  # Equal aspect ratio ensures that pie is drawn as a circle.




    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(9, 5))
    fig.subplots_adjust(wspace=0)

    # pie chart parameters
    overall_ratios = [.13, .04, .83]
    labels = ['Fake', 'Uncertain', 'Genuine']
    explode = [0, 0.1, 0]
    # rotate so that first wedge is split by the x-axis
    wedges, *_ = ax1.pie(overall_ratios, autopct='%1.1f%%', startangle=-60,
                        labels=labels, explode=explode, colors=colour)

    age_ratios = [.72, .13, .06, .09]
    age_labels = ['singleton users', 'singleton products', 'singletone users and products', 'others']
    bottom = 1
    width = .2

    # Adding from the top matches the legend.
    for j, (height, label) in enumerate(reversed([*zip(age_ratios, age_labels)])):
        bottom -= height
        bc = ax2.bar(0, height, width, bottom=bottom, color='C8', label=label,
                    alpha=0.1 + 0.25 * j)
        ax2.bar_label(bc, labels=[f"{height:.0%}"], label_type='center')

    ax2.set_title("node's status")
    ax2.legend(loc=9, bbox_to_anchor=(-0.1, 1.15))
    ax2.axis('off')
    ax2.set_xlim(- 2.5 * width, 2.5 * width)

    # use ConnectionPatch to draw lines between the two plots
    theta1, theta2 = wedges[1].theta1, wedges[1].theta2
    center, r = wedges[1].center, wedges[1].r
    bar_height = sum(age_ratios)

    # draw top connecting line
    x = r * np.cos(np.pi / 180 * theta2) + center[0]
    y = r * np.sin(np.pi / 180 * theta2) + center[1]
    con = ConnectionPatch(xyA=(-width / 2, bar_height), coordsA=ax2.transData,
                        xyB=(x, y), coordsB=ax1.transData)
    con.set_color([0, 0, 0])
    con.set_linewidth(4)
    ax2.add_artist(con)

    # draw bottom connecting line
    x = r * np.cos(np.pi / 180 * theta1) + center[0]
    y = r * np.sin(np.pi / 180 * theta1) + center[1]
    con = ConnectionPatch(xyA=(-width / 2, 0), coordsA=ax2.transData,
                        xyB=(x, y), coordsB=ax1.transData)
    con.set_color([0, 0, 0])
    ax2.add_artist(con)
    con.set_linewidth(4)
    # plt.tight_layout()
    # plt.savefig('reports/paper/pics/reviewDist2.pgf')
    plt.savefig(f'reports/paper/pics/reviewDist2.{saveFormat}')

    plt.show()


if __name__ == '__main__':
    degDist() 
    runningTimes()
    runningTime2()
    convergenceAll()
    convergenceOne()
    fraudDist()
    deviationRange()
    circleConvergenge()
    print(iterationLabel(1, 10))
    catgDist()
    reviewDist()
    reviewDist2()