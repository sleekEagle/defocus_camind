'''
obtain various plots
'''
import numpy as np
import matplotlib.pyplot as plt

'''
plot the camera parameter estimation results
'''

f_10_kcams=np.array([0.98691914, 0.94310617, 1.02744309, 0.89207999, 0.76059744,
       0.96078139, 0.76026974, 1.04522205, 1.02432216, 1.00618229,
       0.84564159, 1.02190048, 0.77301112, 1.02590453, 1.00291503,
       0.88533639, 0.74900771, 0.80254129, 0.93678469, 0.88373789,
       0.76255004, 0.75326266, 0.82642299, 0.95528395, 0.75561235,
       0.86860007, 0.94202093, 0.77646962, 0.86767652, 0.93793033,
       0.98524155, 1.00608831, 0.88144258, 0.89665428, 0.90860334,
       0.91745261, 1.02842111, 0.9067448 , 0.98052404, 0.77998728,
       1.01127167, 0.83109443, 0.93432186, 0.97131169, 0.92444912,
       0.98049246, 0.82976014, 0.94863529, 0.76270578, 0.78062228,
       0.83978526, 0.93475298, 0.98170356, 0.74830805, 0.93061233,
       0.81750522, 0.81636004, 0.95367984, 0.78987792, 0.86911169,
       0.77284625, 0.8316818 , 0.78749252, 0.98064191, 0.95800927,
       0.96116493, 1.01897958, 1.01542215, 1.01728742, 0.83517326,
       0.76819621])

f_20_kcams=np.array([2.13651103, 2.07698962, 1.99833707, 2.18323238, 2.22370663,
       1.99074873, 2.01747691, 2.21885381, 1.97406103, 1.84606768,
       2.13928316, 1.83060697, 1.87417944, 1.87680669, 1.80904097,
       2.11202533, 1.83981337, 2.19651561, 2.08953078, 1.99145534,
       2.18282594, 1.91025432, 2.09159463, 1.89814739, 2.07565047,
       2.18469964, 2.20069477, 2.20190439, 1.8160017 , 2.22761408,
       2.15743973, 1.81013219, 1.87754955, 2.10061748, 2.18923559,
       2.09796624, 2.09647269, 2.1628569 , 1.81064189, 2.14214673,
       2.00470266, 2.13391879, 2.00242767, 2.07055266, 1.80303886,
       2.01751505, 1.82788153, 1.82963892, 1.83833683, 2.19435698,
       2.09543332, 2.20334668, 2.17475814, 2.10653255, 1.92139298,
       2.00319304, 1.81121142, 1.92995838, 2.11271367, 1.83208549,
       2.1893681 , 1.96084805, 2.14101864, 2.12419662, 1.80292529,
       2.04191567, 2.05423564, 2.0714551 , 1.95151944, 1.8695731 ,
       1.80327917, 1.84042404, 1.95197895, 1.90240331, 1.98951227,
       2.08283538, 1.89191974, 1.9614984 , 1.89257863, 1.80512277,
       2.14639752, 1.94014847, 2.01517889, 1.93163895, 2.23054247,
       1.96305143, 2.14938767, 1.94786418, 2.07313452, 2.04170604,
       2.15782278, 2.21400226, 2.04884682, 1.91269123, 2.04987286,
       1.99864634, 2.01470571, 2.15845696, 1.93634874, 1.91387844,
       2.0262862 , 2.01001609, 2.08069572, 2.07578326, 1.99212741,
       1.82111317, 2.02255564, 2.15079859, 2.01587468, 2.10782233,
       1.82553588, 2.15122762, 2.10160222, 2.11375386, 1.86726512,
       1.95474316, 2.19046843, 2.01231672, 2.01329189, 1.86018174,
       1.84438678, 1.82953265, 1.93365176, 2.22303464, 1.84441593,
       1.82878834, 2.04646164, 2.13042122, 2.18204493, 2.04970777,
       1.90781897, 1.98109452, 2.20856296, 2.20108964, 2.01551773,
       1.9110487 , 1.85606968, 2.01143231, 1.94249275, 2.17773538,
       1.94472323, 2.21774253, 2.2148543 , 1.98985589, 2.15283081,
       1.92451374, 2.17051953, 1.9462932 , 2.14062368, 1.98131659,
       1.99748904, 2.14865103, 1.88078562, 1.92986977, 1.8920419 ,
       1.83803967, 1.83292001, 1.89734517, 2.01754656, 2.03407372,
       1.87982685, 1.94196947, 2.11866177, 1.98279458, 2.03474025,
       2.20049906, 2.13375265, 2.2292263 , 1.84406297, 2.15803786,
       1.84386992, 1.93684481, 1.98541877, 2.19177937, 2.20580157,
       2.19669906, 2.12931226, 2.07004797, 1.85711876, 2.15224009,
       1.95575225, 1.89424202, 1.93130859, 1.89281644, 1.96307436,
       2.09315849, 2.14359977, 2.12128815, 2.15801464, 2.13823535,
       1.91632013, 2.02358242, 2.08787602, 2.07561323, 2.15529142,
       2.01291858, 2.0489062 , 2.01695635, 2.12201217, 1.96582675,
       1.81186273, 2.03369922, 1.8698586 , 1.90186898])

f_25_kcams=np.array([3.14024429, 3.20671299, 3.13583621, 3.2187131 , 3.41071931,
       3.27655577, 3.13614119, 3.16145973, 3.28733474, 3.33389863,
       3.25531462, 3.15317194, 3.3752643 , 3.37840214, 3.3531863 ,
       3.17181791, 3.18982485, 3.30480979, 3.12327169, 3.27927886,
       3.23408823, 3.1252063 , 3.24138653, 3.34179224, 3.22061726,
       3.16439733, 3.12301917, 3.23091124, 3.33860192, 3.16420059,
       3.31615222, 3.2322009 , 3.18091484, 3.19843873, 3.43509348,
       3.19245047, 3.4460723 , 3.41549831, 3.17811951, 3.21145975,
       3.24786624, 3.42480743, 3.46496724, 3.30038671, 3.29327225,
       3.3603836 , 3.15203513, 3.42796437, 3.30802765, 3.16239861,
       3.16318238, 3.24956355, 3.13401683, 3.24250088, 3.22719596,
       3.41531291, 3.15403056, 3.13635703, 3.46871224, 3.36632486,
       3.16309791, 3.46251044, 3.21648389, 3.47432337, 3.33563376,
       3.35809008, 3.14727525, 3.38474898, 3.37433362, 3.48566561,
       3.36226787, 3.35945273, 3.25750583, 3.24385235, 3.28153432,
       3.14850386, 3.12085152, 3.1598022 , 3.18358374, 3.22010546,
       3.45848498, 3.3158529 , 3.39507973, 3.22070956, 3.26884161,
       3.14732652, 3.24559012, 3.2003314 , 3.42117518, 3.32467083,
       3.17553267, 3.32271564, 3.33009196, 3.19633652, 3.22436433,
       3.18285174, 3.2264591 , 3.45229667, 3.20175101, 3.13540981,
       3.14927143, 3.47225682, 3.35734854, 3.33163841, 3.29689313,
       3.30208606, 3.13041214, 3.3009915 , 3.4343289 , 3.45182524,
       3.45678482, 3.3811428 , 3.39617413, 3.13963835, 3.42748823,
       3.41906032, 3.29635293, 3.11882039, 3.16603848, 3.12006003,
       3.3205601 , 3.12476116, 3.2823083 , 3.38644835, 3.34219688,
       3.3120285 , 3.31036731, 3.28335243, 3.41049161, 3.17436801,
       3.30682657, 3.12665913, 3.27582906, 3.20391208, 3.48081184,
       3.20095954, 3.25767925, 3.2422749 , 3.1418243 , 3.45190188,
       3.29407295, 3.28785449, 3.25620316, 3.34663628, 3.23594049,
       3.14167868, 3.35556744, 3.42848476, 3.30210816, 3.23092913,
       3.29851536, 3.47599157, 3.25083885, 3.13799656, 3.47139395,
       3.19144805, 3.34009087, 3.17713247, 3.26529619, 3.12876408,
       3.12570393, 3.45510053, 3.15051981, 3.21979061, 3.18021017,
       3.21311805, 3.3519505 , 3.23552662, 3.46441633, 3.10853231,
       3.26612511, 3.28290157, 3.3284033 , 3.10806341, 3.27537052,
       3.48042354, 3.35203979, 3.42425815, 3.33373241, 3.35782091,
       3.35511983, 3.25658268, 3.42652783, 3.30704261, 3.40092904,
       3.20786679, 3.14142064, 3.21673212, 3.14370658, 3.40176173,
       3.31270111, 3.38265711, 3.48638893, 3.40879665, 3.34044756,
       3.15478309, 3.38756549, 3.16084512, 3.36850988, 3.2612347 ,
       3.17420487, 3.24083626, 3.23703955, 3.42403135, 3.15184573,
       3.12113293, 3.29772467, 3.12261375, 3.38468902, 3.14275192,
       3.28622628, 3.39021921, 3.13866542, 3.25827399, 3.22763236,
       3.21271429, 3.13717261, 3.25048001, 3.34413469, 3.45077897,
       3.17105533, 3.23349184, 3.10623352, 3.45532149, 3.31556284,
       3.29961834, 3.42696086, 3.44427401, 3.44571892, 3.39039042,
       3.426848  , 3.42949285, 3.2106496 , 3.38612322, 3.26743946,
       3.26519487, 3.12060501, 3.40144602, 3.45724694, 3.17537895,
       3.21621247, 3.38600965, 3.38969701, 3.25371462, 3.28479363,
       3.21067471, 3.13956621, 3.35257021, 3.33209598, 3.26056885])

f_30_kcams=np.array([4.5320158 , 4.74586224, 4.5427836 , 4.90659395, 4.7791244 ,
       4.86488838, 4.87206358, 4.85240758, 4.79311864, 4.86222106,
       4.78948281, 4.84266273, 4.58647001, 4.53308987, 4.76176739,
       4.79265001, 4.55435697, 4.71112442, 4.72160686, 4.84055024,
       4.76537574, 4.73343892, 4.82694687, 4.92753084, 4.8859311 ,
       4.65307063, 4.61276976, 4.581561  , 4.65922038, 4.63400463,
       4.90610688, 4.79513623, 4.69680472, 4.63443185, 4.75110768,
       4.80929898, 4.63289147, 4.64672911, 4.65352981, 4.56322676,
       4.59300467, 4.81135337, 4.74745419, 4.73435069, 4.75628391,
       4.82736996, 4.71762869, 4.78825577, 4.71417127, 4.86873313,
       4.75936572, 4.92134566, 4.67230679, 4.62622762, 4.56005855,
       4.77004296, 4.92242022, 4.83705586, 4.59339393, 4.52768908,
       4.75368058, 4.55879038, 4.64511627, 4.90327261, 4.91145912,
       4.53260698, 4.79239638, 4.7510575 , 4.66518238, 4.64907927,
       4.81837349, 4.87611734, 4.58949039, 4.78435074, 4.78096524,
       4.68045972, 4.8042247 , 4.85346272, 4.81595875, 4.8402139 ,
       4.87286423, 4.69641254, 4.74435198, 4.59409938, 4.65619415,
       4.55177463, 4.61512846, 4.71421336, 4.6285852 , 4.75810346,
       4.66402795, 4.92001317, 4.58832181, 4.56632082, 4.53163569,
       4.87994102, 4.6481251 , 4.84054461, 4.8380722 , 4.81125149,
       4.7849298 , 4.82379251, 4.76040788, 4.61846271, 4.55956062,
       4.69768008, 4.86794253, 4.77779099, 4.72554076, 4.8082193 ,
       4.89077736, 4.840777  , 4.59176907, 4.85473537, 4.64717854,
       4.57526   , 4.63499536, 4.6993271 , 4.55293441, 4.6688613 ,
       4.7880167 , 4.53192441, 4.83263638, 4.7159963 , 4.6890326 ,
       4.68744555, 4.69146626, 4.83804543, 4.6947045 , 4.74451359,
       4.91612753, 4.73182689, 4.58491792, 4.79457322, 4.58158567,
       4.87475515, 4.74346917, 4.53830275, 4.81688141, 4.57816724,
       4.92494451, 4.71202932, 4.69538131, 4.91019299, 4.71061734,
       4.57025799, 4.7207042 , 4.89311481, 4.64670648, 4.7315564 ,
       4.62442827, 4.64248742, 4.94543656, 4.92433516, 4.83023526,
       4.92453435, 4.7692309 , 4.58893852, 4.68931939, 4.72996755,
       4.67095535, 4.64213312, 4.93102825, 4.77840108, 4.67551539,
       4.54108974, 4.70846897, 4.82413832, 4.61848562, 4.88937786,
       4.86286186, 4.6204252 , 4.82047411, 4.63404911, 4.8258457 ,
       4.92409972, 4.75365892, 4.67293109, 4.68318388, 4.67591887,
       4.7984206 , 4.62521588, 4.73505402, 4.60304118, 4.74359154,
       4.7870761 , 4.54771132, 4.79176455, 4.92669332, 4.81446551,
       4.785871  , 4.85480392, 4.71986415, 4.60777259, 4.68124055,
       4.59211072, 4.56075004, 4.67032303, 4.55827287, 4.53616546,
       4.71153287, 4.56244234, 4.80326377, 4.56811478, 4.82162898,
       4.5707247 , 4.56817911, 4.56158504, 4.70677797, 4.61756788,
       4.78066054, 4.85201824, 4.77248002, 4.81682922, 4.85950959,
       4.87274602, 4.71803634, 4.73461706, 4.56232448, 4.62068976,
       4.75733697, 4.56537806, 4.8242385 , 4.70489579, 4.6052713 ,
       4.68156788])

f_40_kcams=np.array([8.62450861, 8.45906904, 8.43512069, 8.12282008, 8.17797643,
       8.62903639, 8.11320773, 8.38971618, 8.40039042, 8.3831161 ,
       8.50588149, 8.28378332, 8.41761502, 8.56881065, 8.22309962,
       8.26918153, 8.2770678 , 8.14582287, 8.12270027, 8.21502766,
       8.4876597 , 8.3582079 , 8.57008333, 8.45723881, 8.53101475,
       8.24741864, 8.21253722, 8.24657328, 8.29489354, 8.29024448,
       8.25542638, 8.31538862, 8.28747757, 8.51488881, 8.25511872,
       8.39744204, 8.53763067, 8.232463  , 8.27507946, 8.46308153,
       8.20052362, 8.56622758, 8.16249737, 8.13072647, 8.61009047,
       8.26434229, 8.33022997, 8.16250099, 8.46719123, 8.32740987,
       8.1758414 , 8.42634483, 8.17898633, 8.36045574, 8.39530414,
       8.30308914, 8.39240222, 8.27084117, 8.47996173, 8.12924641,
       8.44607132, 8.42451539, 8.17094367, 8.27239001, 8.12321647,
       8.23752446, 8.28609085, 8.26765931, 8.38706707, 8.43463082,
       8.5410279 , 8.58828775, 8.50856878, 8.52849603, 8.27144598,
       8.25069377, 8.13578312, 8.14425459, 8.28406449, 8.13163271,
       8.35191139, 8.36771958, 8.26129506, 8.45951165, 8.60492375,
       8.49009045, 8.12483767, 8.51797955, 8.11305881, 8.16807905,
       8.18844217, 8.3769291 , 8.46058284, 8.40164384, 8.43126463,
       8.41408248, 8.10651847, 8.18061052, 8.41098715, 8.38391972,
       8.30760039, 8.3333928 , 8.18973791, 8.21704494, 8.61836249,
       8.15061982, 8.12018479, 8.52922265, 8.18358035, 8.38966595,
       8.16616829, 8.2922359 , 8.36571248, 8.60520541, 8.27517069,
       8.33414364, 8.28048819, 8.35317962, 8.40909168, 8.32740764,
       8.26243716, 8.32124527, 8.44767886, 8.47304309, 8.24347643,
       8.37483757, 8.44047695, 8.28516481, 8.43380592, 8.55412722,
       8.5583011 , 8.61241139, 8.62863904, 8.4980886 , 8.50295362,
       8.32380623, 8.44391838, 8.28792315, 8.16614609, 8.49426926,
       8.13358247, 8.55039631, 8.46591399, 8.31133193, 8.50837288,
       8.20091812, 8.3333381 , 8.62120353, 8.29799337, 8.35318451,
       8.51665788, 8.22440641, 8.16185688, 8.20974815, 8.12172849,
       8.23790499, 8.256624  , 8.45224732, 8.51582881, 8.16877751,
       8.34115909, 8.44277407, 8.47372373, 8.35110192, 8.33587659,
       8.19554534, 8.28917724, 8.27729982, 8.23448844, 8.15142001,
       8.38513849, 8.3747793 , 8.25216553, 8.29553177, 8.51821484,
       8.23531813, 8.27706469, 8.16628772, 8.56445721, 8.17980792,
       8.39736521, 8.23874327, 8.58091957, 8.53676133, 8.27150098,
       8.20431537, 8.29634268, 8.31396657, 8.24549214, 8.2993733 ,
       8.19527382, 8.20929186, 8.31711707, 8.54525931])

f_50_kcams=np.array([12.84061875, 12.86787878, 12.61890009, 12.65971836, 12.7217698 ,
       12.71424763, 12.36507032, 12.83943755, 12.37461059, 12.91419548,
       12.66281799, 12.97586765, 12.78684006, 12.76548809, 12.68549657,
       12.61445122, 12.69783526, 12.63564337, 12.71885909, 12.5317691 ,
       12.4270279 , 12.64741046, 12.77166204, 12.66783421, 12.73617382,
       12.55769944, 12.41024005, 12.53469238, 12.38805266, 12.50102856,
       12.63858758, 12.84704289, 12.85227105, 12.60794465, 12.74952416,
       12.80550773, 12.54115747, 12.3770301 , 13.03319368, 12.52222611,
       12.64856824, 12.45017459, 13.03857888, 12.57848628, 12.59707115,
       13.04608605, 12.80462122, 12.41638633, 12.36941406, 12.41713045,
       12.69810149, 12.9659775 , 12.43814757, 12.54215464, 12.70448261,
       12.83945472, 13.00656526, 12.86120546, 12.67442874, 12.82953522,
       13.07025762, 12.70294719, 12.83004386, 12.94551254, 13.01895973,
       12.47107023, 12.69139462, 12.5389119 , 12.7515156 , 12.50947642,
       12.44497522, 12.63614992, 13.07198921, 12.76097616, 12.57851849,
       12.65163625, 12.91482205, 12.51595646, 12.9328684 , 12.74890737,
       12.45107934, 12.41210898, 12.49869474, 12.99814732, 12.93505317,
       13.03365929, 12.86830433, 12.57568579, 12.95582227, 12.75149449,
       12.3499027 , 12.60986649, 12.47327881, 13.02809368, 12.75530272,
       12.61491669, 12.43287043, 12.47710293, 12.69763723, 12.73629351,
       12.42387384, 13.02464805, 12.51834984, 12.4333555 , 12.61856204,
       12.44855882, 12.68538877, 12.41829333, 13.0575868 , 12.66782353,
       12.36689239, 12.85577155, 12.82268858, 12.57387822, 12.44002993,
       12.88727019, 12.35629426, 12.40350325, 12.58877846, 12.77272833,
       12.38235476, 12.44640659, 12.44824684, 12.43966504, 12.75077012,
       12.58548939, 12.49774975, 13.03168287, 12.39352516, 12.55233451,
       12.5870058 , 12.39377221, 12.97659003, 12.80208086, 12.82368045,
       12.41383644, 12.39458221, 12.46487725, 12.9296425 , 12.56507003,
       12.46164925, 12.72021554, 12.6553557 , 12.36428469, 12.42915626,
       12.90178278, 12.82383608, 12.73362759, 12.54411082, 12.4629413 ,
       12.49210925, 12.85791862, 12.57597046, 12.58126855, 12.95865605,
       12.60731568, 12.97248684, 12.81444692, 12.54306715, 12.47401161,
       12.37155706, 12.38581393, 12.71177135, 12.92636432, 12.66910392,
       13.04646609, 12.97908317, 12.79501904, 12.47373548, 12.59776261,
       12.48915114, 12.53096195, 12.71222168, 12.43529554, 12.43518583,
       12.70679341, 12.85745855, 12.66036696, 12.93327843, 12.74977183,
       12.41727003, 12.83378661, 12.68071379, 12.41466178, 13.02142127,
       12.57708929, 12.41466381, 12.38949453, 12.89599211, 12.35137002,
       12.40124497, 12.95513421, 13.00933711, 12.49519956, 12.52812262,
       12.78446646, 12.53697682, 12.61606203, 12.81925901, 12.48083877,
       12.45943095, 12.84678395, 12.69092234, 12.63363989, 12.99088069])

#obtain scaled GT kcam calues to compare to the estimated values
fdist=2
N=1
px=36*1e-6
f=np.array([10,20,25,30,40,50])
kcam_GT=(f*1e-3)**2/(fdist-f*1e-3)/N/px
k_cams_est=np.array([0.89,2.01,3.28,4.72,8.34,12.66])
plt.plot(kcam_GT,k_cams_est)
plt.show()

#estimate all the camera parameters given two GT kcams at f=10 and f=50 and the kcam_est values at f=10 and f=50
gt_est = (kcam_GT[-1]-kcam_GT[0])/(k_cams_est[-1]-k_cams_est[0])*(k_cams_est-k_cams_est[0]) + kcam_GT[0]
plt.plot(kcam_GT)
plt.plot(gt_est)
plt.show()

#scale all estimated kcam values like above
f_10_kcams_scaled=(kcam_GT[-1]-kcam_GT[0])/(k_cams_est[-1]-k_cams_est[0])*(f_10_kcams-k_cams_est[0]) + kcam_GT[0]
f_20_kcams_scaled=(kcam_GT[-1]-kcam_GT[0])/(k_cams_est[-1]-k_cams_est[0])*(f_20_kcams-k_cams_est[0]) + kcam_GT[0]
f_25_kcams_scaled=(kcam_GT[-1]-kcam_GT[0])/(k_cams_est[-1]-k_cams_est[0])*(f_25_kcams-k_cams_est[0]) + kcam_GT[0]
f_30_kcams_scaled=(kcam_GT[-1]-kcam_GT[0])/(k_cams_est[-1]-k_cams_est[0])*(f_30_kcams-k_cams_est[0]) + kcam_GT[0]
f_40_kcams_scaled=(kcam_GT[-1]-kcam_GT[0])/(k_cams_est[-1]-k_cams_est[0])*(f_40_kcams-k_cams_est[0]) + kcam_GT[0]
f_50_kcams_scaled=(kcam_GT[-1]-kcam_GT[0])/(k_cams_est[-1]-k_cams_est[0])*(f_50_kcams-k_cams_est[0]) + kcam_GT[0]



# from sklearn.linear_model import RANSACRegressor
# kcam_GT=np.expand_dims(kcam_GT,1)
# reg = RANSACRegressor(random_state=0).fit(kcam_GT, k_cams_est)
# reg.estimator_.coef_
# reg.estimator_.intercept_

kcam_GT_=kcam_GT
fig, ax = plt.subplots(figsize=(12, 7))
ax.boxplot(f_10_kcams_scaled, positions=[round(kcam_GT[0],1)])
ax.boxplot(f_20_kcams_scaled, positions=[round(kcam_GT_[1],1)])
ax.boxplot(f_25_kcams_scaled, positions=[round(kcam_GT_[2],1)])
ax.boxplot(f_30_kcams_scaled, positions=[round(kcam_GT_[3],1)])
ax.boxplot(f_40_kcams_scaled, positions=[round(kcam_GT_[4],1)])
ax.boxplot(f_50_kcams_scaled, positions=[round(kcam_GT[5],1)])
ax.plot(kcam_GT_,kcam_GT_,'bo-',linewidth=0.5,markersize=3)
plt.savefig("G:\\My Drive\\focus-defocus\\camind\\images\\kcamest\\total.png",dpi=500)
plt.show()

#seperate box plots
kcam_GT_=kcam_GT
i=5
boxprops=dict(linestyle='-', linewidth=5)
medianprops=dict(linestyle='-', linewidth=5)
whiskerprops=dict(linestyle='-', linewidth=5)
capprops=dict(linestyle='-', linewidth=5)

fig, ax = plt.subplots(figsize=(12, 7))
ax.boxplot(f_50_kcams_scaled, positions=[round(kcam_GT[i],2)],boxprops=boxprops,medianprops=medianprops,whiskerprops=whiskerprops,capprops=capprops)
plt.ylim([np.min(f_50_kcams_scaled)-0.5,np.max(f_50_kcams_scaled)+0.5])
ax.plot(round(kcam_GT_[i],2),round(kcam_GT_[i],2),'bo-',markersize=15)
ax.set_facecolor("#e6fce6")

plt.savefig("G:\\My Drive\\focus-defocus\\camind\\images\\kcamest\\"+str(i)+'.png',dpi=300)
plt.show()


ax.boxplot(f_20_kcams_scaled, positions=[round(kcam_GT_[1],1)])
ax.boxplot(f_25_kcams_scaled, positions=[round(kcam_GT_[2],1)])
ax.boxplot(f_30_kcams_scaled, positions=[round(kcam_GT_[3],1)])
ax.boxplot(f_40_kcams_scaled, positions=[round(kcam_GT_[4],1)])
ax.boxplot(f_50_kcams_scaled, positions=[round(kcam_GT[5],1)])
ax.plot(kcam_GT_,kcam_GT_,'bo-')
plt.show()


'''
NYU estimating kcam values

----experiments----
trained models used 1/kcam scaled with f_base 
We do not use N,px for these kcams
these values are proportional to base_f** (s1-f)/f**2

----theory-----
theory was developped for kcam (not scaled)
these values are proportional to f**2/(s1-f/N/px)

When we estimate kcam with the blur calibration we estimate the theoratical value. 
We need to trasform this to experimetal values. 
'''

base_f=25e-3
fdist=2
N=1
px=36*1e-6

#estimated kcams (from circualr pattern experiments)
k_cams_est=np.array([0.89,2.01,3.28,4.72,8.34,12.66])

#thoeratical GT kcams
f=np.array([10,20,25,30,40,50])
kcam_GT_exp=(f*1e-3)**2/(base_f**2)/(fdist-f*1e-3)

#scale estimated kcams to 1/kcam scaled with f_base
# we need to do this because expperimantal estimation plot has an intercept and need to avoid it. 
gt_est = (kcam_GT_exp[-1]-kcam_GT_exp[0])/(k_cams_est[-1]-k_cams_est[0])*(k_cams_est-k_cams_est[0]) + kcam_GT_exp[0]
est_kcams=1/gt_est
kcam_exp_GT=1/kcam_GT_exp
#use est_kcams and kcam_exp_GT as kcams in the experiments (to evaluate the networks)

plt.plot(kcam_exp_GT,est_kcams)
plt.show()


#stimate experiment kcam for a given kcam from defocus blur calibration
kcam_calib=8.24
gt_est = (kcam_GT_exp[-1]-kcam_GT_exp[0])/(k_cams_est[-1]-k_cams_est[0])*(kcam_calib-k_cams_est[0]) + kcam_GT_exp[0]
kcam_exp=1/gt_est

#convert exp kcam to GT
f=np.array([10,20,25,30,40,50])*1e-3
kcam_GT=f**2/N/(fdist-f)/px
kcam_exp=1/(f**2/base_f**2/(fdist-f))
(1/kcam_exp)/N/px*base_f**2

k=0.5
kcam_GT=(1/k)/N/px*base_f**2




#get kcam values consistant with the theory
kcams_GT_theory = kcam_GT_exp*N/px*base_f**2
kcams_est_therory = (kcams_GT_theory[-1]-kcams_GT_theory[0])/(k_cams_est[-1]-k_cams_est[0])*(k_cams_est-k_cams_est[0]) + kcams_GT_theory[0]
plt.plot(kcams_GT_theory,kcams_est_therory)
plt.show()



'''
RMSE vs kcam used
'''
from scipy.interpolate import make_interp_spline
#f=30 
#kcam values used in the experiments
kcams_30_theory=np.arange(kcams_GT_theory[3]-5,kcams_GT_theory[3]+5,step=0.5)
kcam_30_exp=1/(kcams_30_theory/N*px/base_f**2)
rmse_30=[0.1488,0.1459,0.1428,0.1394,0.1358,0.1321,0.1286,0.1256,0.1236,0.1227,0.1232,0.1250,0.1281,0.1324,0.1377,0.1437,0.1505,0.1577,0.1652,0.1729,0.1809]

kcams_40_theory=np.arange(kcams_GT_theory[4]-5,kcams_GT_theory[4]+5,step=0.5)
kcam_40_exp=1/(kcams_40_theory/N*px/base_f**2)
rmse_40=[0.1385,0.1326,0.1276,0.1237,0.1207,0.1187,0.1174,0.1169,0.1168,0.1171,0.1178,0.1186,0.1195,0.1206,0.1218,0.1231,0.1244,0.1258,0.1271,0.1285]

X_Y_Spline_30 = make_interp_spline(kcams_30_theory,rmse_30)
X_Y_Spline_40 = make_interp_spline(kcams_40_theory, rmse_40)

kcams_30_=np.arange(np.min(kcams_30_theory),np.max(kcams_30_theory),step=0.01)
kcams_40_=np.arange(np.min(kcams_40_theory),np.max(kcams_40_theory),step=0.01)

Y30_=X_Y_Spline_30(kcams_30_)
Y40_=X_Y_Spline_40(kcams_40_)

kcams_30_theory_=np.append(kcams_30_theory,[12.69,12.53])
rmse_30_=rmse_30 +[0.1232,0.1229]
ind=np.argsort(kcams_30_theory_)
kcams_30_theory_=kcams_30_theory_[ind]
rmse_30_=np.array(rmse_30_)[ind]

kcams_40_theory_=np.append(kcams_40_theory,[22.67,23.05])
rmse_40_=rmse_40 +[0.1178,0.1183]
ind=np.argsort(kcams_40_theory_)
kcams_40_theory_=kcams_40_theory_[ind]
rmse_40_=np.array(rmse_40_)[ind]

plt.plot(kcams_30_theory_,rmse_30_,'ro-',markersize=4)
plt.plot(kcams_40_theory_,rmse_40_,'bo-',markersize=4)

# plt.plot(kcams_30_,Y30_,'ro',linewidth=3)
# plt.plot(kcams_40_,Y40_,'b0',linewidth=3)

#***for f=30****
#plot GT kcam performance

# >>> kcam_exp_GT
# array([12.4375    ,  3.09375   ,  1.975     ,  1.36805556,  0.765625  ,
#         0.4875    ])
# >>> est_kcams
# array([12.4375    ,  3.73210369,  2.08070558,  1.38555315,  0.75306852,
#         0.4875    ])

plt.plot([12.69],[0.1232],marker="*", markersize=20, markerfacecolor="#00fbff",markeredgewidth=0)
#plot estimated kcam performance
plt.plot([12.53],[0.1229],marker="+", markersize=20, markeredgecolor="#092d8c")

#***for f=40****
#plot GT kcam performance
plt.plot([22.67],[0.1178],marker="*", markersize=20, markerfacecolor="#ff00b2",markeredgewidth=0)
#plot estimated kcam performance
plt.plot([23.05],[0.1183],marker="+", markersize=20, markeredgecolor="#7b0632")

plt.xlim([11.0,24.0])
plt.ylim([0.115,0.13])
plt.xlabel('kcam used')
plt.ylabel('RMSE')
plt.savefig(r'G:\My Drive\focus-defocus\camind\images\kcamest\est_error_variation_mag.png',dpi=300)
plt.show()


import cv2
path=r'C:\Users\***\Downloads\f_10_focused.jpg'
img=cv2.imread(path,cv2.IMREAD_GRAYSCALE)
values=img[54,:]*-1
# plt.plot(values)
# plt.show()

kernel_size = 10
kernel = np.ones(kernel_size) / kernel_size
data_convolved = np.convolve(values, kernel, mode='same')[int(kernel_size/2):]
plt.plot(data_convolved,linewidth=5)
plt.axis('off')
plt.show()


'''
Various blur (sigma) vs distance plots for different cameras
'''

s1=2
N=1
px=36*1e-6
f=np.array([25,50,30,40])*1e-3
kcams=f**2/N/px/(s1-f)
s2=np.arange(0.1,5,0.01)
fig, ax = plt.subplots()

for f_ in f:
    sigma=np.abs(s1-s2)/s2/(s1-f_)*f_**2/N/px
    ax.plot(s2,sigma)
plt.xlabel('Distance in m',fontsize='x-large')
plt.ylabel(r'$\sigma$',fontsize='x-large')
plt.ylim([0,15])
legend = ax.legend(['f=10mm' , 'f=20mm' , 'f=35mm'],loc='upper center', fontsize='x-large')
plt.savefig(r'C:\Users\***\Downloads\sig_blur.png',dpi=300,bbox_inches='tight')
plt.show()


'''
blender dataset kcam calculation
'''
fdist=1.5
N=2
px=36*1e-6
f=np.array([3,4,5,6])
kcam_GT=(f*1e-3)**2/(fdist-f*1e-3)/N/px


'''
calculate the kcam of defocusnet dataset
'''

N=1
px=36*1e-6
f=2.9e-3
fdist=np.array([0.1,.15,.3,0.7,1.5])
kcams=f**2/(fdist-f)/N/px

N=2
px=36*1e-6
f=np.array([3,4,5,6])*1e-3
fdist=1.5
kcams_test=f**2/(fdist-f)/N/px


'''
plot blur weight vs RMSE
'''
from scipy.interpolate import make_interp_spline

bw=[0.1,0.3,0.5,0.8,1.0,1.5]
rmse_30=[0.123,0.113, 0.123,0.124,0.123,0.125]
rmse_40=[0.097,0.101,0.101,0.105,0.117,0.097]

#plot splines
X_Y_Spline_30 = make_interp_spline(bw,rmse_30,k=2)
X_Y_Spline_40 = make_interp_spline(bw, rmse_40,k=2)

spline_x=np.arange(np.min(bw),np.max(bw),step=0.001)

Y30_=X_Y_Spline_30(spline_x)
Y40_=X_Y_Spline_40(spline_x)

plt.plot(spline_x,Y30_)
plt.plot(spline_x,Y40_)
plt.plot(bw,rmse_30,'bo')
plt.plot(bw,rmse_40,'ro')
plt.savefig(r'G:\My Drive\focus-defocus\camind\images\bweigth.png',dpi=300,bbox_inches='tight')






