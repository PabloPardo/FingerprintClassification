#include "FingerPrintClassification.h"
#include <iostream>
#include "opencv2\highgui\highgui.hpp"
#include "opencv2\ml\ml.hpp"
#include <Windows.h>

const bool CROSS = true;

struct InputCross
{
	char* csvPath;
	char* normalizedDataPath;
	char* outPutPath;
	Properties* prop;
};

struct Input
{
	char* csvPath;
	char* imagesPath;
	char* outPutPath;
	Properties* prop;
};

InputCross getRandomizedData() {
	InputCross ret = InputCross();
	ret.csvPath = "\\\\ssd2015\\Data\\CSVs\\20150907\\RandomizedData.csv";
	ret.normalizedDataPath = "\\\\ssd2015\\Data\\CSVs\\20150907\\features_full_python.csv";
	ret.outPutPath = "\\\\ssd2015\\Data\\out\\PythonNormalizedData\\model\\";
	Properties* prop = new Properties();
	prop->n_bins = 32;
	prop->rad_grad = 1;
	prop->rad_dens = 3;
	prop->rad_entr = 5;
	prop->max_depth = 25;
	prop->min_samples_count = 2; 
	prop->max_categories = 3;
	prop->max_num_of_trees_in_forest = 100;
	prop->verbose = true;
	ret.prop = prop;
	return ret;
}

Input getSegmented_Depth16() {
	Input ret = Input();
	ret.csvPath = "\\\\ssd2015\\data\\out\\Segmented_Depth16\\CSVs\\fitData0.csv";
	ret.imagesPath = "\\\\ssd2015\\data\\Segmented\\";
	ret.outPutPath = "\\\\ssd2015\\data\\out\\Segmented_Depth16\\model0\\";
	Properties* prop = new Properties();
	prop->n_bins = 32;
	prop->rad_grad = 1;
	prop->rad_dens = 3;
	prop->rad_entr = 5;
	prop->max_depth = 25;
	prop->min_samples_count = 2; 
	prop->max_categories = 3;
	prop->max_num_of_trees_in_forest = 100;
	prop->verbose = true;
	ret.prop = prop;
	return ret;
}

Input getSegmentedWholeData() {
	Input ret = Input();
	ret.csvPath = "\\\\ssd2015\\data\\CSVs\\RandomizedData.csv";
	ret.imagesPath = "\\\\ssd2015\\data\\Segmented\\";
	ret.outPutPath = "\\\\ssd2015\\data\\out\\All_Segmented_Depth25\\model\\";
	Properties* prop = new Properties();
	prop->n_bins = 32;
	prop->rad_grad = 1;
	prop->rad_dens = 3;
	prop->rad_entr = 5;
	prop->max_depth = 25;
	prop->min_samples_count = 2; 
	prop->max_categories = 3;
	prop->max_num_of_trees_in_forest = 100;
	prop->verbose = true;
	ret.prop = prop;
	return ret;
}

Input getSegmentedData() {
	Input ret = Input();
	ret.csvPath = "\\\\ssd2015\\data\\CSVs\\RandomizedData.csv";
	ret.imagesPath = "\\\\ssd2015\\data\\Segmented\\";
	ret.outPutPath = "\\\\ssd2015\\data\\CSVs\\";
	Properties* prop = new Properties();
	prop->n_bins = 32;
	prop->rad_grad = 1;
	prop->rad_dens = 3;
	prop->rad_entr = 5;
	prop->max_depth = 25;
	prop->min_samples_count = 2; 
	prop->max_categories = 3;
	prop->max_num_of_trees_in_forest = 100;
	prop->verbose = true;
	ret.prop = prop;
	return ret;
}

InputCross getProva() {
	InputCross ret = InputCross();
	ret.csvPath = "\\\\ssd2015\\Data\\CSVs\\20150907\\RandomizedData.csv";
	ret.normalizedDataPath = "\\\\ssd2015\\Data\\CSVs\\20150907\\trainFeaturesData.csv";
	ret.outPutPath = "\\\\ssd2015\\Data\\out\\PythonNormalizedData2\\model\\";
	Properties* prop = new Properties();
	prop->n_bins = 32;
	prop->rad_grad = 1;
	prop->rad_dens = 3;
	prop->rad_entr = 5;
	prop->max_depth = 25;
	prop->min_samples_count = 2; 
	prop->max_categories = 3;
	prop->max_num_of_trees_in_forest = 100;
	prop->nactive_vars = 0;
	prop->verbose = true;
	ret.prop = prop;
	return ret;
}

void FitAndPredict(void)
{
	float* probs;
	ReturnType ret;

	try
	{
		if(CROSS)
		{
			InputCross paths = getProva();
			ret = SetProperties(paths.prop);

			std::cout << "Train ...\n";
			
			ret = FitFromDataRF(paths.csvPath,
				paths.normalizedDataPath,
				paths.outPutPath);
			if(ret.code > 0)
				throw std::exception(ret.message,ret.code);
				
			std::cout << "Test ...\n";
			
			double normalizedFeatures[914] = { 1.845062416057740434e-02,-1.952805605716775514e-02,1.454429534864720985e-01,-1.806450013907548724e-01,4.366603085119361016e-01,-5.748491832562102610e-01,5.674110866107644480e-01,-6.025982534742019281e-01,1.064602781158302314e+00,-3.715322877141474245e-01,-1.732201984932369432e-01,-1.109322896259168356e-01,-1.143207677782540621e-01,-1.461030280960289029e-01,-1.670936476136786475e-01,-2.088130837231358949e-01,-2.510410644957181137e-01,-2.843849023460253700e-01,-4.254319037944505610e-01,-5.709501886557051265e-01,-4.220097878483975995e-01,-3.865686846894345607e-01,-2.671818495626717138e-01,-1.646944517659507112e-01,-1.567646744813129545e-01,-1.325099691018284953e-01,-9.223222567335109812e-02,-4.171585797573328092e-02,-4.183823111628842684e-02,-4.616251917591864679e-02,-1.024480284543337061e-01,-9.236504272493045731e-02,-3.347465887700858400e-01,-3.077849901891092088e-01,-3.133405549246440214e-01,-3.014915646359205281e-01,3.820704371855961901e-01,-7.783439321519671772e-01,-3.030334559681028828e-01,-2.990435855359544481e-01,-2.971922202426884319e-01,-2.972613239640798377e-01,-2.979037006356654582e-01,-7.759357067191699486e-02,-8.834625677685956080e-02,-9.075337792788591706e-02,-1.153224643238521113e-01,-1.300589824818806084e-01,-5.549794340460004749e-01,1.303210378072930942e+00,-3.794541128760884341e-01,-2.381904148951395073e-01,-1.810195897615829441e-01,-2.064689279270627287e-01,-2.076098371527282105e-01,-3.390352298307971468e-01,-1.489183378192353024e-01,6.094085073582128365e-02,1.740403310345372268e-01,9.533049167314287808e-02,-1.125593603753055755e-01,-3.426974580926431813e-01,-3.854121875819853660e-01,-5.836524837013544742e-01,-1.163946400450667357e-01,6.211067743833552646e-01,4.322256914891688395e-01,4.440233307494471648e-01,4.476864060070507856e-01,4.716914559360261827e-01,2.395897994020615429e-01,-7.980732447673719032e-02,2.200725455681167564e-01,3.932383754188388814e-01,1.506814931299177984e-01,1.193799651522612776e-01,7.245693499391568926e-01,8.773082989256801856e-01,8.964340653674053305e-01,9.489746575457543853e-01,1.215950548490708050e+00,1.087054233075307907e+00,1.037379973083271789e+00,9.238316264328801930e-01,7.116142207075154547e-01,4.436690006986802604e-01,2.994772873029100113e-01,-7.251685733815397238e-03,-1.756624020563245792e-01,-2.858802720871963521e-01,-4.144695917891432768e-01,-5.006755941481288996e-01,-4.413031783339169123e-01,-2.061831121868125394e-01,-3.045625080484223868e-01,-2.616579515602243666e-01,1.610507632869771166e+00,1.770721502840914008e-01,-1.052231207080112432e-02,3.845385599271738952e-01,3.310576007469300253e-01,-2.265603010152585917e-01,-5.537928119325623566e-01,3.988089843319806566e-02,-3.049219109552946394e-01,-3.717513665167967996e-01,-7.053112196055016936e-01,-4.898005551810836744e-01,-9.674386706313512496e-01,-8.329059797578762803e-01,-6.406498858271724162e-01,-9.636384540827986900e-01,-6.624971514605519918e-01,-1.109298861834398631e+00,-7.855285365894084659e-01,-1.074020934846992681e+00,-1.165874387560444037e+00,-9.069386288443842536e-01,-4.172577453292331540e-01,-9.131615797089190911e-01,-7.984757043473256211e-01,-7.877891171398559989e-01,-1.051656876299672527e+00,-1.893412594391957748e-01,-2.435574040189388834e-01,-8.336304702156613899e-01,-1.201972353401707050e+00,-1.515663804848087770e-01,-1.253083373353682806e+00,6.940034346341333871e-01,-1.154093540092195475e-01,-3.980114397507409918e-01,-4.375853762077952513e-01,-4.723742629537808213e-01,-3.269870219041596249e-01,-3.081778227063059483e-01,2.476815606441329712e-02,3.393074967385759783e-02,1.088388713583144513e-01,2.908535057252186196e-01,-7.039095050258577235e-02,-1.814169354344384866e-01,-1.007045502887677496e-01,6.475346325140483028e-01,1.214764317383102699e-01,1.364875928268544314e+00,-6.728370451796248375e-01,1.262215437713277488e+00,-2.308785773422945276e-01,1.912790743884922318e+00,-5.637006542376012952e-01,-4.111065947792627573e-01,-3.259623441755092887e-01,-2.668565889490907561e-01,-2.276516482987529788e-01,-2.023845301348995429e-01,-1.872340746028683389e-01,-1.759926868610796058e-01,-1.620563551253910162e-01,-1.560284147140438282e-01,-1.518576490740700669e-01,-1.501672497067717271e-01,-1.324178462557865366e-01,-1.303499465829608406e-01,-1.281102257824430490e-01,-1.194017404437867913e-01,-1.016355080628429941e-01,-9.847853042459826789e-02,-7.742302884456940537e-02,-9.133485075481545534e-02,-1.019375659068098294e-01,-1.356020964960075181e-01,-1.788824415121866573e-01,1.106568320094531543e+00,-1.869403824681722670e-01,-1.049661509483136024e-01,1.950672393933643167e-01,1.530187807242617826e+00,-1.020870174317347168e+00,-3.028513539380295327e-01,-2.134948992345672147e-01,-2.403252985711645251e-01,-2.145140115749445409e-01,-7.430364660810094835e-01,-6.455165902801838218e-01,-1.577538370797394035e-02,-7.334464716643304683e-02,-2.193812780412772445e-01,3.324917975011234983e+00,-4.213897427644814897e-01,-6.238860575861020497e-01,-5.298060726252590280e-01,-3.765906785604785090e-01,-2.659082052731323031e-01,-2.060676965950412054e-01,-1.717238785554439373e-01,-1.509495901921336647e-01,-1.306859204620637316e-01,-1.183735751131755470e-01,-1.019344420906104498e-01,-8.271756530589595424e-02,-6.131504643327341519e-02,-6.665549672269556469e-02,-1.282889549984584654e-01,-1.901249100115788293e-01,7.078080111646474215e-01,5.382581927593338422e-01,-2.182662731175913362e-02,6.909239776590589410e-02,-5.709630316271716022e-02,-4.921301184990422573e-01,-1.000117778422356940e+00,-8.022166210991029667e-01,-6.965591379258851967e-01,-3.935283290900768449e-01,-4.436183562015089121e-01,-3.862050194927881819e-01,-2.353914209908126165e-01,-2.201440732006996748e-01,-6.209710914438590912e-01,-8.067123165273768493e-01,-1.949562412986942661e-01,-7.520991654846701513e-01,-6.948883470533288031e-01,-6.369087734480778362e-01,-6.022157538840944291e-01,-6.037464962140669522e-01,-6.228510875259625212e-01,-5.808600959202293579e-01,-6.091978502623769387e-01,-6.835126504484289045e-01,-7.066914665872563450e-01,-6.321692390154101115e-01,-5.336953951282310538e-01,-4.873584500985878587e-01,-3.498299311719956228e-01,-3.101213688985787753e-01,-6.877474075523306585e-01,3.431745586452762953e-01,-8.228888339408203523e-01,-8.537206594570379004e-01,-8.549354550418252208e-01,-1.744288525983295202e-01,-5.385838030524544573e-01,-5.499438314741660339e-01,-5.626133056218364237e-01,-8.568266391057154374e-01,-8.406863249475744482e-01,-5.755916551982963592e-01,-5.780172658699789245e-01,-5.908429600054228015e-01,-5.950405980583051146e-01,-3.318313762853258853e-01,-5.900849875150070423e-01,-3.202336631889050045e-01,-5.772358198814712127e-01,-8.314424635332149371e-01,-5.483347097859391228e-01,-2.309428908966983940e-01,-4.910340495417244155e-01,-7.985614874530417850e-01,-7.961981971968087324e-01,-8.141360131196373473e-01,-3.162173034770765501e-01,-7.953121318448916899e-01,-7.805741139524636774e-01,7.045793469630603489e-01,-7.349405852979954412e-01,4.878915814950772756e-01,-1.660846054626707036e-01,-1.509442881210940057e-01,5.912679891226289008e-01,1.126117095704108478e-01,1.144498412961641343e+00,-5.506094532820018372e-01,1.127137646382294633e+00,-2.513111213046868575e-01,1.578675576385450530e+00,-4.625425929891291976e-01,-3.438511395499484480e-01,-2.617418578979068267e-01,-2.055629341354416051e-01,-1.686577132367610976e-01,-1.490955957042933433e-01,-1.296737998333950259e-01,-1.378903954392734776e-01,-1.150666146545909901e-01,-1.169964650494769304e-01,-1.204162649025148218e-01,-1.204298022634854165e-01,-1.274937922254012479e-01,-1.482544005413956034e-01,-1.894294004325483560e-01,-1.748361733659671846e-01,-1.501979336785572161e-01,-1.512250113441389876e-01,-1.274126355579358327e-01,-9.863571858101066736e-02,-1.096855635684538810e-01,-1.338069919408048369e-01,-1.724795710521322689e-01,1.156833720926564224e+00,-2.007559541625689725e-01,-1.094554586434323407e-01,1.616792530783187631e-01,1.248571737935320725e+00,-8.254869214363338425e-01,-1.463543690406930720e-01,-7.384690188157730373e-02,-1.973281102198002346e-01,-2.015188676416185543e-01,-2.121255194187067830e-02,-1.766871938268723963e-01,-2.172254571994369082e-02,-6.001735567165576019e-02,-2.398206441303866454e-01,3.016522727925632807e+00,-3.233977344102473839e-01,-6.078393967581965374e-01,-4.988388151873562881e-01,-3.255009587642374580e-01,-2.210078447488896813e-01,-1.779966023821540344e-01,-1.539653636643724122e-01,-1.408273984151786340e-01,-1.326734907198337532e-01,-1.489499099412771588e-01,-1.330553013113847105e-01,-9.318587750086713972e-02,-6.807284828011544997e-02,-6.755308900180735654e-02,-1.211341070065346387e-01,-1.832612130470287481e-01,-4.454690418740356872e-01,6.852655789923760565e-01,4.721082529033012731e-01,4.974177844717476149e-01,4.597475666073800316e-01,6.574130989949930060e-01,4.993237586537386252e-01,6.483515081027949600e-01,1.194029514451117324e-01,4.232491234360658550e-01,1.716881374664532556e-01,2.676763209166349577e-01,6.242430697871751422e-01,5.258474617803059470e-01,4.587334286295795538e-01,7.170933386534213971e-01,1.237340029165393818e-01,1.343003421787518192e+00,1.311712247135818465e+00,1.433399391772494313e+00,1.334417372186402551e+00,1.004872460347170593e+00,8.651909690918831997e-01,3.646849015580854725e-01,-2.299554920749785625e-02,-8.333369442641253011e-02,-1.307992010355790019e-02,-4.947449805815724855e-02,-7.533372879298695646e-02,-2.361413642396722745e-01,-3.208625836258715558e-01,5.556488908810798544e-02,1.788878921652998014e-02,1.623480246061991705e+00,1.839219457954130166e-01,5.020792068104786843e-01,3.322421665919353395e-01,2.158847165559044379e-01,1.202099541037591202e+00,3.700681373151367137e-01,1.785238990083760191e+00,-6.785758474188552380e-02,4.152376504176888417e-01,1.065674196358195047e-01,2.878256905889366446e-01,-2.202929778970563468e-01,1.894444927974938964e-01,-4.901859613434849661e-01,-9.182100100967788281e-01,-3.023074288795885578e-01,-3.174076825279628089e-01,-3.364716800390665785e-01,8.719626307822722078e-01,-5.381053266815668534e-01,-3.222594300248698418e-01,-2.925046560157130160e-01,-9.338372101677998582e-01,-9.275606352587489889e-01,-9.238665990159711239e-01,-8.970204153770544586e-01,-8.718204660620239022e-01,-8.434492138302214226e-01,-7.933774263251279768e-01,1.749090667254913911e-01,-3.910478386476822221e-01,-8.845126347579412851e-02,-2.557670546760270475e-02,-1.548748533937785998e-01,8.553666712249513582e-01,-6.590164853984030735e-01,9.047095465274308523e-01,-4.197556040338094596e-01,1.496716736667168757e+00,-5.946327017975180373e-01,-3.776978796976837582e-01,-3.249115730900828947e-01,-3.293753021751281151e-01,-3.615433135188880209e-01,-4.028954625240210308e-01,-5.075661247049044755e-01,-4.205688489969406918e-01,-5.892643358389559882e-01,-5.113348308696022970e-01,-5.383852790564467306e-01,-3.498065538576467515e-01,-1.669600617728100034e-01,1.411894087013626509e-01,4.245678054633081344e-01,2.401151973426735031e-01,2.946464338972809571e-01,1.629537112607711280e-01,2.633401603998120066e-01,1.344522896416568136e-01,2.577522313167435186e-01,1.218682131931417834e-01,-7.363849764817608987e-03,1.159061836632849118e+00,-1.544399658145960141e-01,-1.418612087450072667e-01,2.118438798857933542e-02,1.032758023527168056e+00,-7.464165262082782526e-01,-3.865496765903260190e-01,-3.344760390049050791e-01,-8.155869459282916178e-01,-6.521024554345871138e-01,-7.050089959120319216e-01,-8.384984240543792433e-01,-4.533315372203092913e-02,-1.036408986175591240e-01,-3.283269585036828198e-01,2.539101169945157555e+00,-4.659172420049963459e-01,-6.547266620903406942e-01,-5.425535186495106243e-01,-4.388243449796846996e-01,-4.294483781480618600e-01,-3.993873963188088738e-01,-2.696438719210635937e-01,-1.266008199212240448e-01,-3.135995085027530882e-01,-4.247090008877968392e-01,-9.174832168092147411e-01,-8.435816124212127676e-01,-7.133458363128964974e-01,-7.274384230216598413e-01,-3.873669734227950801e-01,-1.461858792954221220e-01,8.916489475577294987e-01,3.275946861775632102e-01,1.393120672586929398e-01,1.989535266609130382e-01,3.200323110048385322e-01,6.018852945763545348e-01,2.974820774794897371e-01,-4.311177658978337934e-01,8.692250970364756579e-01,1.353845879884332248e+00,1.744036205888504609e+00,6.547875658839279467e-01,1.226746984703803278e+00,6.157999907442106569e-01,1.186328545682455715e-01,-2.367952113968446459e-01,-5.775104877319230390e-01,-7.520283389520228123e-01,-6.484979181898687939e-01,-6.689328891876342720e-01,-6.881396535915872414e-01,-7.097444788533040505e-01,-8.033613865403933518e-01,-8.240554810871062452e-01,-8.966896478006897153e-01,-9.886056342383453543e-01,-1.013547461993465015e+00,-9.469406976226594219e-01,-3.238033286751612616e-01,3.949241065999518630e-01,-1.456275479981654908e-01,-4.676066913618099341e-01,-3.020298787687212827e-01,-4.428762492570600950e-01,-1.085005224331103690e+00,-1.269355173968274464e+00,-8.285424871920445966e-01,-6.687808047606875750e-01,-7.913637305294323188e-01,-8.941426159904809490e-01,-2.881933816347144184e-01,-1.042712101582570883e+00,-4.595872826881871598e-01,-9.530572901258301810e-01,-1.613602296055209884e+00,-7.843578795984337826e-02,-7.210010929594897755e-01,-9.139458191708152990e-01,-9.503978896258448161e-01,-9.374278380428923141e-01,-1.108823194942597512e+00,-8.825724200397108143e-01,-1.133822902348191164e+00,-1.486199652076923394e+00,-1.659625052694905678e+00,-1.026403085696400819e+00,-9.035992734059402620e-01,-7.894470153866843809e-01,-6.239731063251663379e-01,-4.478352907608326716e-01,1.418314260566027329e-01,-8.526909796258770324e-01,-6.313034356136266734e-01,4.278428590169841006e-01,-3.846561050951570748e-01,-1.250989779714781758e-01,-9.135107532610638181e-02,-1.671320415383931446e-01,5.095747637436559563e-01,-2.714781821535224848e-01,6.188520598207086776e-01,-4.463496958129749559e-01,8.511644936710502929e-01,-4.457249575430408985e-03,8.071814679907193812e-02,2.540510210579766896e-03,-5.941848116586170153e-02,-1.575951469745198730e-01,-2.041578936503351815e-01,-3.257578337534543511e-01,-3.793699404582089318e-01,-4.726088124552567660e-01,-5.800953359381167962e-01,-6.512311519767240986e-01,-7.592632343194727351e-01,-6.351204298303955165e-01,-5.031547149658627260e-01,-5.118212625079845424e-01,-3.510808853216207392e-01,-3.581131682386662907e-01,-2.209527779679657244e-01,-1.679545440911761889e-01,-4.981717597422640614e-02,9.697632449358396800e-02,-1.055845389876035184e-01,-6.962531843235214568e-02,-2.795142209033089076e-01,-1.624315478090496423e-01,-1.648450610120687221e-01,-3.681346225228058697e-03,5.944991612870100806e-01,-2.816274157231797304e-01,4.469935481013141698e-02,-5.367954479460496781e-02,-3.108089806137176447e-01,-6.875647903676571637e-01,-7.278224244384400299e-01,-5.907539041675361480e-01,-5.837914878828082133e-02,-1.090638438331451160e-01,-3.359064626860521763e-01,-5.494397090633704739e-01,2.001303745087193331e+00,-4.180759053348917509e-01,-3.962063373756089524e-01,-3.093906927667030571e-01,-3.359866789954000232e-01,-3.709059548095465786e-01,-3.352934662561832257e-01,-5.688256247707080826e-01,-6.448526346895552264e-01,-7.607982247764564221e-01,-8.176042018234966902e-01,-7.893068784378278746e-01,-7.625861208688712312e-01,-6.603035005376227673e-01,-1.397839002472738568e-01,-1.750639436472617161e-01,-7.983153784073593329e-01,9.096677179621471820e-01,7.755317728782669118e-01,7.568181052977978052e-01,7.052000638991736681e-01,1.899624578999282998e-01,-2.187492862056607257e-01,-6.902843728573700988e-01,-8.379534121284758497e-01,-8.982788171439508584e-01,-1.165013123857603139e+00,-7.118242550418985015e-01,-6.512571807750848762e-01,3.704876536850700286e-01,1.187722992105584963e+00,1.771216822264769508e+00,2.789493965352680238e+00,3.043063025601428606e+00,3.184339351089951364e+00,2.926161929927393768e+00,2.315533410204580367e+00,1.554610240460934767e+00,1.065599368242268641e+00,4.919127733886639042e-01,1.173176729683117758e-01,-2.451585676702665861e-01,-4.866672003131806146e-01,-7.267004963512155857e-01,-8.629747073911897548e-01,-4.104458483977797556e-01,-2.804989824199573545e-01,-1.933473412921817014e-01,-7.836349258218908154e-02,1.397193089457580806e+00,-8.528371319507659365e-01,-1.052293747778838134e-02,2.004833877232913952e+00,-2.467144880540458884e-01,-4.618913578227555883e-01,3.824864453144815601e-01,-3.426325206069327778e-01,-6.978593222625277537e-02,-2.108302521086899070e-02,2.195865958744474822e-01,-1.134178258244153414e-01,-5.641895616745131736e-01,-3.125509252725139353e-01,-2.018491541626699914e-01,5.565354776152445920e-02,-7.000655228401814201e-01,3.084495701652839594e-01,-5.197003070476949871e-01,1.965232979728054052e-01,-4.479212329096180500e-01,-9.410265326085451298e-02,8.736117544722071804e-01,-5.758914577922810141e-01,-6.495527058859973391e-01,-1.665454259932753978e+00,-1.125251257207687150e+00,-3.772845132324275097e-01,-1.118431462545049548e+00,-7.270850629703310419e-01,1.249332270147473611e+00,4.233223074457949375e-02,-7.793113175462577924e-02,6.730508858704519204e-01,8.718233682346386570e-02,1.222937697172710170e+00,-7.190344717696676158e-01,1.116347529544203354e+00,-2.453494295121227686e-01,1.813193235706795026e+00,-5.736439226489434828e-01,-4.095201498041294852e-01,-3.213871210165467063e-01,-2.718956033774507786e-01,-2.293835072508567663e-01,-2.026988422424839953e-01,-1.661443235919912997e-01,-1.472229974739512059e-01,-1.241497319093137486e-01,-1.107397089642487842e-01,-1.061955609430219744e-01,-1.009712035127035279e-01,-1.141400097027676364e-01,-1.169240146209401393e-01,-9.508229884263570686e-02,-1.169167017383394985e-01,-9.973986689254225879e-02,-9.623936955400871907e-02,-7.905046640721806461e-02,-8.213728145094449651e-02,-7.467681382793554812e-02,-8.270236109571033056e-02,-1.108091946763790914e-01,1.213662711400671945e+00,-1.560534526635699581e-01,-1.091703739178028210e-01,6.247898631159207877e-02,1.517293599707489005e+00,-1.154541221043338561e+00,-3.012654110696713072e-01,-2.980129937105516560e-02,-1.550008109108562127e-01,2.645468026184942534e-02,-2.160911991285509037e-01,-4.397929079820369624e-01,-1.256249689496555938e-02,-6.864274946108120390e-02,-2.211209638589569648e-01,-3.319080787663692966e-01,2.674802148580379679e+00,-5.226145633263091783e-01,-5.739226511775421535e-01,-4.319777756473414665e-01,-2.780852140539592510e-01,-2.021466556562240557e-01,-1.640801631975992991e-01,-1.048421683222390821e-01,-9.637101258417087035e-02,-9.529911788507057502e-02,-1.585908468640169489e-01,-8.829605456630823523e-02,-7.020056006039848695e-02,-7.358295744858509013e-02,-8.423560500492424663e-02,-1.158269271491380453e-01,1.224566701944828223e-01,2.564949236421492640e-01,3.617493420065185927e-01,4.720064755173680182e-01,1.933257406107548593e-01,5.353125819152854126e-01,1.037249589924435916e-01,8.712884061624927290e-02,4.946768884067462846e-01,6.087417768066514245e-01,5.106809090356937419e-01,3.738119011343038478e-01,3.408108719497015549e-01,6.729470080131395859e-02,-1.481703277476839331e-01,-4.994408646402472007e-02,-5.168684952388243048e-02,-4.839380125538279009e-01,-6.292928429138369140e-01,-5.593100371959004447e-01,-4.807731318942294885e-01,-4.435970646069340440e-01,-3.477557812673499060e-01,-2.599845406476445353e-01,-2.059188357954508941e-01,1.445730464261463233e-01,1.291921032166586103e-01,2.983620105711104822e-02,-1.293312989967569104e-01,-1.991343153530257259e-01,-2.838023544763320083e-01,-3.649765629962254398e-01,-1.239775938070144107e-01,-8.467049241534869708e-01,-4.276789525646397805e-01,-5.043770969451192299e-01,-1.958374563290582182e-01,-6.286259069566394464e-01,-3.175213259092286577e-01,-9.858740471497429159e-01,-7.106802184275697876e-01,-4.525067659283871779e-01,-4.742725955351040357e-01,1.252605551463420973e-02,-7.587685567734757663e-01,-7.711380362254137699e-01,-3.283837804964006568e-01,-3.414784343490538943e-01,-3.593142128871961161e-01,-1.002698425121347103e+00,4.555252071617474363e-02,3.639464108105196910e-02,2.284568967558589514e-02,-5.970798541335389231e-01,-8.172106405940444684e-01,-8.214554705060521655e-01,-3.800890320199263317e-01,-1.053565699202528849e+00,-1.067906969540795226e+00,-1.087188883658931626e+00,-3.494254446314803975e-01,2.566052989809684548e-01,-4.414278712969990082e-01,-8.313455010444081861e-01,2.738567490550166891e-02,-1.037178748116571042e-01,6.211614875299126259e-01,-4.718362618086452409e-01,-9.975157500420902545e-01,1.464937409719865746e+00,-9.651827168706570426e-01,-2.533014004001126196e-01,-7.109359973157206580e-01,1.184136023226753043e+00,3.380233125792034832e-01,-1.239295765223113466e-01,-9.640472087749248076e-02,-7.707566783396729848e-02,-9.257846645708762978e-02,-4.888090126148498388e-02,-5.592114319632972980e-02,-9.050948761796077935e-02,-9.097998061013685966e-02,-1.206284244801383299e-01,-1.325419269929050681e-01,-2.070935864403673221e-01,-1.766182512205748678e-01,-1.921119575984805983e-01,-1.928144567809861598e-01,-2.342156305529959071e-01,-1.902438947507353040e-01,-1.660719297232354785e-01,-1.289113926379670816e-01,-1.023769623525614264e-01,-9.989799839038074336e-02,-1.025329482459568786e-01,5.879622349285489047e-02,-1.508419056369966482e-01,-1.342352852318607348e-01,-2.719952667278958680e-01,-7.929541495849561361e-01,7.677398031976141457e-01,9.884993338011989483e-02,1.194384451748228271e-01,-1.286841213450249655e-01,-1.161638290104413523e-01,8.519799952522995046e-01,1.569688984790229558e+00,-1.082454638103540027e-02,-5.605508775608302602e-02,-2.270610701537479625e-01,-3.593918815844943659e-01,-4.672549894098319556e-01,-6.462442208628266238e-01,1.545404623496383412e+00,-5.741812988009822177e-02,-1.177225775761279458e-01,-4.923098078538919176e-02,7.435872455418812525e-03,9.245041943705854803e-02,8.893394487622446887e-02,8.631291991402804953e-02,8.882884450406808297e-02,4.275925902708463400e-02,-2.634282885381056954e-02,-5.173131739917690569e-02,-7.122943888168492554e-02,-1.063543907335534333e-01,-8.452477143344770960e-01,7.592792853554992671e-01,6.900547115134926246e-01,7.938312246125016625e-01,7.596642058614035964e-01,5.755865681438033787e-01,9.003125066579441560e-01,1.543678304589377326e-01,4.724440259393792862e-01,2.409289037838852066e-01,7.930858184223169016e-02,3.582316961149943912e-01,1.054579867779843916e+00,2.052503755382098394e+00,2.277555345707966250e+00,2.310084247856370343e+00,4.640219656748619248e-01,1.865148097077307110e+00,1.785423295421227463e+00,1.471108812321022308e+00,1.258179851557659434e+00,1.161349829449091375e+00,1.112740882156059152e+00,9.520594670280566607e-01,9.076210400065343187e-01,7.337087124759728018e-01,5.604467323717096106e-01,7.010489364488099728e-02,-1.818624747051553470e-01,-5.194399804961933320e-01,-5.552434545465823046e-01,-4.677748266538377941e-01,4.599821149937371317e-02,4.723601892233360577e-01,2.002720944012938709e-01,-5.172320878500797914e-01,-1.473340127990016013e-01,5.607795049086278238e-01,9.241624919095275947e-03,5.855690875046213328e-01,1.991236227048360785e+00,3.159247366208884356e-01,4.793995986052347336e-01,6.216006285375896834e-01,4.459741735438284976e-02,-4.573273345914045662e-01,-7.336993447329318019e-02,5.114662083376372292e-01,5.804274582574894120e-02,-1.619044283445882637e-01,-9.037857865087841613e-01,-1.994537158673585775e-01,-7.316425766275294063e-01,-5.394208407690750384e-01,2.034854434026082612e-02,9.651936487379267904e-01,2.982017991368458021e-01,9.970818748197198822e-01,7.795142583017893034e-02,-2.802706774040587545e-01,-1.575166175117256318e-01,3.610299508948811575e-01,3.192538617078725105e-01,2.416906695159359852e-01,6.089459334109257466e-02,-8.697182862906982503e-01,-7.108547422946431826e-01,1.308857753783522759e-01,-4.610402507277306938e-01};
			void* handle;
			ret = InitModel(&handle,paths.outPutPath);
			ret = CrossPredictRF(&probs, 
				handle, normalizedFeatures);
			ret = ReleaseModel(handle);
			
			if(ret.code > 0)
				throw std::exception(ret.message,ret.code);
		}
		else
		{
			Input paths = getSegmentedData();
			
			ret = SetProperties(paths.prop);
			
			std::cout << "Train ...\n";

			ret = FitRF(paths.csvPath,
					paths.imagesPath, paths.outPutPath);
			
			if(ret.code > 0)
				throw std::exception(ret.message,ret.code);
		
			std::cout << "Test ...\n";

			char* imagePath = "\\\\ssd2015\\data\\Segmented\\2014-11-27_7941300_10.png";
			int features[13] = {9, 4, 1165, 101, 98, 86, 55, 51, 9, 9, 9, 8, 6};
			cv::Mat in = cv::imread(imagePath,cv::IMREAD_GRAYSCALE);
			void* handle;
			ret = InitModel(&handle,paths.outPutPath);
			ret = PredictRF(&probs, in.data, in.rows, in.cols,paths.outPutPath,handle,features);
			ret = ReleaseModel(handle);
			if(ret.code > 0)
				throw std::exception(ret.message,ret.code);
		}

		std::cout << "Borrosa:" << probs[0] << std::endl;
		std::cout << "Petita:" << probs[1] << std::endl;
		std::cout << "Negre:" << probs[2] << std::endl;
		std::cout << "Clara" << probs[3] << std::endl;
		std::cout << "Motejada" << probs[4] << std::endl;
		std::cout << "Defectuosa" << probs[5] << std::endl;
		
		ret = ReleaseFloatPointer(probs);
	}
	catch(std::exception& ex)
	{
		std::cout << ex.what() << std::endl;
	}
	system("pause");
}

Input getPredictFP() {
	Input ret = Input();
	ret.csvPath = "\\\\ssd2015\\data\\CSVs\\Malos_15_07_08.csv";
	ret.imagesPath = "\\\\ssd2015\\data\\PredictData_Segmented\\";
	ret.outPutPath = "\\\\ssd2015\\data\\out\\Malos_15_07_08_Segmented\\";
	Properties* prop = new Properties();
	prop->n_bins = 32;
	prop->rad_grad = 1;
	prop->rad_dens = 3;
	prop->rad_entr = 5;
	prop->max_depth = 25;
	prop->min_samples_count = 2; 
	prop->max_categories = 3;
	prop->max_num_of_trees_in_forest = 100;
	prop->verbose = true;
	ret.prop = prop;
	return ret;
}

void ExtractFeatures(void) 
{
	Input input = getSegmentedData();
	SetProperties(input.prop);
	ExtractFeatures(input.csvPath,input.imagesPath,input.outPutPath);
}

void ExportNormalizationVector()
{
	const char* unNormalizedDataPath = "//ssd2015/Data/Normalization_Comparison/features_full_python_unnormalized.csv";
	ExportMeanStdFile(unNormalizedDataPath, unNormalizedDataPath,true);
}


int main(void){

	//FitAndPredict();
	ExtractFeatures();
	//ExportNormalizationVector();
	return 0;
}