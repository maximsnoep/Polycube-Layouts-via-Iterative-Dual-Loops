use crate::{
    doconeli::Doconeli,
    solution::{Evaluation, PrincipalDirection, Surface},
    ColorType, Configuration,
};
use bevy::{
    prelude::*,
    render::{mesh::Indices, render_resource::PrimitiveTopology},
    utils::Instant,
};
use itertools::Itertools;
use rand::Rng;
use std::error::Error;
use std::io::Write;
use std::{f32::consts::PI, ops::Add, path::PathBuf};

pub fn average<'a, T>(list: impl Iterator<Item = T>) -> T
where
    T: Add<Output = T> + std::default::Default + std::ops::Div<f32, Output = T>,
{
    let (sum, count) = list.fold((T::default(), 0.), |(sum, count), elem| {
        (sum + elem, count + 1.)
    });
    sum / (count as f32)
}

pub fn set_intersection<T: std::cmp::PartialEq + Clone>(
    collection_a: &Vec<T>,
    collection_b: &Vec<T>,
) -> Vec<T> {
    let mut intesection = collection_b.clone();
    intesection.retain(|edge_id| collection_a.contains(edge_id));
    return intesection;
}

pub fn transform_coordinates(translation: Vec3, scale: f32, position: Vec3) -> Vec3 {
    (position * scale) + translation
}

pub fn intersection_in_sequence(elem_a: usize, elem_b: usize, sequence: &Vec<usize>) -> bool {
    let mut sequence_copy = sequence.clone();
    sequence_copy.retain(|&elem| elem == elem_a || elem == elem_b);
    debug_assert!(sequence_copy.len() == 4, "{:?}", sequence_copy);
    sequence_copy.dedup();
    sequence_copy.len() >= 4
}

// Report times
#[derive(Clone)]
pub struct Timer {
    pub start: Instant,
}

impl Timer {
    pub fn new() -> Timer {
        Timer {
            start: Instant::now(),
        }
    }

    pub fn reset(&mut self) {
        self.start = Instant::now();
    }

    pub fn report(&self, note: &str) {
        // info!("{:>12?}  >  {note}", self.start.elapsed());
    }
}

// X,      Y,       Z
// red,    blue,    yellow
// green,  orange,  purple
pub fn get_color(dir: PrincipalDirection, primary: bool, configuration: &Configuration) -> Color {
    match (dir, primary) {
        (PrincipalDirection::X, true) => configuration.color_primary1.into(),
        (PrincipalDirection::Y, true) => configuration.color_primary2.into(),
        (PrincipalDirection::Z, true) => configuration.color_primary3.into(),
        (PrincipalDirection::X, false) => configuration.color_secondary1.into(),
        (PrincipalDirection::Y, false) => configuration.color_secondary2.into(),
        (PrincipalDirection::Z, false) => configuration.color_secondary3.into(),
    }
}

pub fn get_random_color() -> Color {
    let hue = rand::thread_rng().gen_range(0.0..360.0);
    let sat = rand::thread_rng().gen_range(0.6..0.8);
    let lit = rand::thread_rng().gen_range(0.6..0.8);
    Color::hsl(hue, sat, lit)
}

// Magma colormap: https://github.com/BIDS/colormap/blob/master/option_a.py
pub const MAGMA: [[f32; 3]; 256] = [
    [1.46159096e-03, 4.66127766e-04, 1.38655200e-02],
    [2.25764007e-03, 1.29495431e-03, 1.83311461e-02],
    [3.27943222e-03, 2.30452991e-03, 2.37083291e-02],
    [4.51230222e-03, 3.49037666e-03, 2.99647059e-02],
    [5.94976987e-03, 4.84285000e-03, 3.71296695e-02],
    [7.58798550e-03, 6.35613622e-03, 4.49730774e-02],
    [9.42604390e-03, 8.02185006e-03, 5.28443561e-02],
    [1.14654337e-02, 9.82831486e-03, 6.07496380e-02],
    [1.37075706e-02, 1.17705913e-02, 6.86665843e-02],
    [1.61557566e-02, 1.38404966e-02, 7.66026660e-02],
    [1.88153670e-02, 1.60262753e-02, 8.45844897e-02],
    [2.16919340e-02, 1.83201254e-02, 9.26101050e-02],
    [2.47917814e-02, 2.07147875e-02, 1.00675555e-01],
    [2.81228154e-02, 2.32009284e-02, 1.08786954e-01],
    [3.16955304e-02, 2.57651161e-02, 1.16964722e-01],
    [3.55204468e-02, 2.83974570e-02, 1.25209396e-01],
    [3.96084872e-02, 3.10895652e-02, 1.33515085e-01],
    [4.38295350e-02, 3.38299885e-02, 1.41886249e-01],
    [4.80616391e-02, 3.66066101e-02, 1.50326989e-01],
    [5.23204388e-02, 3.94066020e-02, 1.58841025e-01],
    [5.66148978e-02, 4.21598925e-02, 1.67445592e-01],
    [6.09493930e-02, 4.47944924e-02, 1.76128834e-01],
    [6.53301801e-02, 4.73177796e-02, 1.84891506e-01],
    [6.97637296e-02, 4.97264666e-02, 1.93735088e-01],
    [7.42565152e-02, 5.20167766e-02, 2.02660374e-01],
    [7.88150034e-02, 5.41844801e-02, 2.11667355e-01],
    [8.34456313e-02, 5.62249365e-02, 2.20755099e-01],
    [8.81547730e-02, 5.81331465e-02, 2.29921611e-01],
    [9.29486914e-02, 5.99038167e-02, 2.39163669e-01],
    [9.78334770e-02, 6.15314414e-02, 2.48476662e-01],
    [1.02814972e-01, 6.30104053e-02, 2.57854400e-01],
    [1.07898679e-01, 6.43351102e-02, 2.67288933e-01],
    [1.13094451e-01, 6.54920358e-02, 2.76783978e-01],
    [1.18405035e-01, 6.64791593e-02, 2.86320656e-01],
    [1.23832651e-01, 6.72946449e-02, 2.95879431e-01],
    [1.29380192e-01, 6.79349264e-02, 3.05442931e-01],
    [1.35053322e-01, 6.83912798e-02, 3.14999890e-01],
    [1.40857952e-01, 6.86540710e-02, 3.24537640e-01],
    [1.46785234e-01, 6.87382323e-02, 3.34011109e-01],
    [1.52839217e-01, 6.86368599e-02, 3.43404450e-01],
    [1.59017511e-01, 6.83540225e-02, 3.52688028e-01],
    [1.65308131e-01, 6.79108689e-02, 3.61816426e-01],
    [1.71713033e-01, 6.73053260e-02, 3.70770827e-01],
    [1.78211730e-01, 6.65758073e-02, 3.79497161e-01],
    [1.84800877e-01, 6.57324381e-02, 3.87972507e-01],
    [1.91459745e-01, 6.48183312e-02, 3.96151969e-01],
    [1.98176877e-01, 6.38624166e-02, 4.04008953e-01],
    [2.04934882e-01, 6.29066192e-02, 4.11514273e-01],
    [2.11718061e-01, 6.19917876e-02, 4.18646741e-01],
    [2.18511590e-01, 6.11584918e-02, 4.25391816e-01],
    [2.25302032e-01, 6.04451843e-02, 4.31741767e-01],
    [2.32076515e-01, 5.98886855e-02, 4.37694665e-01],
    [2.38825991e-01, 5.95170384e-02, 4.43255999e-01],
    [2.45543175e-01, 5.93524384e-02, 4.48435938e-01],
    [2.52220252e-01, 5.94147119e-02, 4.53247729e-01],
    [2.58857304e-01, 5.97055998e-02, 4.57709924e-01],
    [2.65446744e-01, 6.02368754e-02, 4.61840297e-01],
    [2.71994089e-01, 6.09935552e-02, 4.65660375e-01],
    [2.78493300e-01, 6.19778136e-02, 4.69190328e-01],
    [2.84951097e-01, 6.31676261e-02, 4.72450879e-01],
    [2.91365817e-01, 6.45534486e-02, 4.75462193e-01],
    [2.97740413e-01, 6.61170432e-02, 4.78243482e-01],
    [3.04080941e-01, 6.78353452e-02, 4.80811572e-01],
    [3.10382027e-01, 6.97024767e-02, 4.83186340e-01],
    [3.16654235e-01, 7.16895272e-02, 4.85380429e-01],
    [3.22899126e-01, 7.37819504e-02, 4.87408399e-01],
    [3.29114038e-01, 7.59715081e-02, 4.89286796e-01],
    [3.35307503e-01, 7.82361045e-02, 4.91024144e-01],
    [3.41481725e-01, 8.05635079e-02, 4.92631321e-01],
    [3.47635742e-01, 8.29463512e-02, 4.94120923e-01],
    [3.53773161e-01, 8.53726329e-02, 4.95501096e-01],
    [3.59897941e-01, 8.78311772e-02, 4.96778331e-01],
    [3.66011928e-01, 9.03143031e-02, 4.97959963e-01],
    [3.72116205e-01, 9.28159917e-02, 4.99053326e-01],
    [3.78210547e-01, 9.53322947e-02, 5.00066568e-01],
    [3.84299445e-01, 9.78549106e-02, 5.01001964e-01],
    [3.90384361e-01, 1.00379466e-01, 5.01864236e-01],
    [3.96466670e-01, 1.02902194e-01, 5.02657590e-01],
    [4.02547663e-01, 1.05419865e-01, 5.03385761e-01],
    [4.08628505e-01, 1.07929771e-01, 5.04052118e-01],
    [4.14708664e-01, 1.10431177e-01, 5.04661843e-01],
    [4.20791157e-01, 1.12920210e-01, 5.05214935e-01],
    [4.26876965e-01, 1.15395258e-01, 5.05713602e-01],
    [4.32967001e-01, 1.17854987e-01, 5.06159754e-01],
    [4.39062114e-01, 1.20298314e-01, 5.06555026e-01],
    [4.45163096e-01, 1.22724371e-01, 5.06900806e-01],
    [4.51270678e-01, 1.25132484e-01, 5.07198258e-01],
    [4.57385535e-01, 1.27522145e-01, 5.07448336e-01],
    [4.63508291e-01, 1.29892998e-01, 5.07651812e-01],
    [4.69639514e-01, 1.32244819e-01, 5.07809282e-01],
    [4.75779723e-01, 1.34577500e-01, 5.07921193e-01],
    [4.81928997e-01, 1.36891390e-01, 5.07988509e-01],
    [4.88088169e-01, 1.39186217e-01, 5.08010737e-01],
    [4.94257673e-01, 1.41462106e-01, 5.07987836e-01],
    [5.00437834e-01, 1.43719323e-01, 5.07919772e-01],
    [5.06628929e-01, 1.45958202e-01, 5.07806420e-01],
    [5.12831195e-01, 1.48179144e-01, 5.07647570e-01],
    [5.19044825e-01, 1.50382611e-01, 5.07442938e-01],
    [5.25269968e-01, 1.52569121e-01, 5.07192172e-01],
    [5.31506735e-01, 1.54739247e-01, 5.06894860e-01],
    [5.37755194e-01, 1.56893613e-01, 5.06550538e-01],
    [5.44015371e-01, 1.59032895e-01, 5.06158696e-01],
    [5.50287252e-01, 1.61157816e-01, 5.05718782e-01],
    [5.56570783e-01, 1.63269149e-01, 5.05230210e-01],
    [5.62865867e-01, 1.65367714e-01, 5.04692365e-01],
    [5.69172368e-01, 1.67454379e-01, 5.04104606e-01],
    [5.75490107e-01, 1.69530062e-01, 5.03466273e-01],
    [5.81818864e-01, 1.71595728e-01, 5.02776690e-01],
    [5.88158375e-01, 1.73652392e-01, 5.02035167e-01],
    [5.94508337e-01, 1.75701122e-01, 5.01241011e-01],
    [6.00868399e-01, 1.77743036e-01, 5.00393522e-01],
    [6.07238169e-01, 1.79779309e-01, 4.99491999e-01],
    [6.13617209e-01, 1.81811170e-01, 4.98535746e-01],
    [6.20005032e-01, 1.83839907e-01, 4.97524075e-01],
    [6.26401108e-01, 1.85866869e-01, 4.96456304e-01],
    [6.32804854e-01, 1.87893468e-01, 4.95331769e-01],
    [6.39215638e-01, 1.89921182e-01, 4.94149821e-01],
    [6.45632778e-01, 1.91951556e-01, 4.92909832e-01],
    [6.52055535e-01, 1.93986210e-01, 4.91611196e-01],
    [6.58483116e-01, 1.96026835e-01, 4.90253338e-01],
    [6.64914668e-01, 1.98075202e-01, 4.88835712e-01],
    [6.71349279e-01, 2.00133166e-01, 4.87357807e-01],
    [6.77785975e-01, 2.02202663e-01, 4.85819154e-01],
    [6.84223712e-01, 2.04285721e-01, 4.84219325e-01],
    [6.90661380e-01, 2.06384461e-01, 4.82557941e-01],
    [6.97097796e-01, 2.08501100e-01, 4.80834678e-01],
    [7.03531700e-01, 2.10637956e-01, 4.79049270e-01],
    [7.09961888e-01, 2.12797337e-01, 4.77201121e-01],
    [7.16387038e-01, 2.14981693e-01, 4.75289780e-01],
    [7.22805451e-01, 2.17193831e-01, 4.73315708e-01],
    [7.29215521e-01, 2.19436516e-01, 4.71278924e-01],
    [7.35615545e-01, 2.21712634e-01, 4.69179541e-01],
    [7.42003713e-01, 2.24025196e-01, 4.67017774e-01],
    [7.48378107e-01, 2.26377345e-01, 4.64793954e-01],
    [7.54736692e-01, 2.28772352e-01, 4.62508534e-01],
    [7.61077312e-01, 2.31213625e-01, 4.60162106e-01],
    [7.67397681e-01, 2.33704708e-01, 4.57755411e-01],
    [7.73695380e-01, 2.36249283e-01, 4.55289354e-01],
    [7.79967847e-01, 2.38851170e-01, 4.52765022e-01],
    [7.86212372e-01, 2.41514325e-01, 4.50183695e-01],
    [7.92426972e-01, 2.44242250e-01, 4.47543155e-01],
    [7.98607760e-01, 2.47039798e-01, 4.44848441e-01],
    [8.04751511e-01, 2.49911350e-01, 4.42101615e-01],
    [8.10854841e-01, 2.52861399e-01, 4.39304963e-01],
    [8.16914186e-01, 2.55894550e-01, 4.36461074e-01],
    [8.22925797e-01, 2.59015505e-01, 4.33572874e-01],
    [8.28885740e-01, 2.62229049e-01, 4.30643647e-01],
    [8.34790818e-01, 2.65539703e-01, 4.27671352e-01],
    [8.40635680e-01, 2.68952874e-01, 4.24665620e-01],
    [8.46415804e-01, 2.72473491e-01, 4.21631064e-01],
    [8.52126490e-01, 2.76106469e-01, 4.18572767e-01],
    [8.57762870e-01, 2.79856666e-01, 4.15496319e-01],
    [8.63320397e-01, 2.83729003e-01, 4.12402889e-01],
    [8.68793368e-01, 2.87728205e-01, 4.09303002e-01],
    [8.74176342e-01, 2.91858679e-01, 4.06205397e-01],
    [8.79463944e-01, 2.96124596e-01, 4.03118034e-01],
    [8.84650824e-01, 3.00530090e-01, 4.00047060e-01],
    [8.89731418e-01, 3.05078817e-01, 3.97001559e-01],
    [8.94700194e-01, 3.09773445e-01, 3.93994634e-01],
    [8.99551884e-01, 3.14616425e-01, 3.91036674e-01],
    [9.04281297e-01, 3.19609981e-01, 3.88136889e-01],
    [9.08883524e-01, 3.24755126e-01, 3.85308008e-01],
    [9.13354091e-01, 3.30051947e-01, 3.82563414e-01],
    [9.17688852e-01, 3.35500068e-01, 3.79915138e-01],
    [9.21884187e-01, 3.41098112e-01, 3.77375977e-01],
    [9.25937102e-01, 3.46843685e-01, 3.74959077e-01],
    [9.29845090e-01, 3.52733817e-01, 3.72676513e-01],
    [9.33606454e-01, 3.58764377e-01, 3.70540883e-01],
    [9.37220874e-01, 3.64929312e-01, 3.68566525e-01],
    [9.40687443e-01, 3.71224168e-01, 3.66761699e-01],
    [9.44006448e-01, 3.77642889e-01, 3.65136328e-01],
    [9.47179528e-01, 3.84177874e-01, 3.63701130e-01],
    [9.50210150e-01, 3.90819546e-01, 3.62467694e-01],
    [9.53099077e-01, 3.97562894e-01, 3.61438431e-01],
    [9.55849237e-01, 4.04400213e-01, 3.60619076e-01],
    [9.58464079e-01, 4.11323666e-01, 3.60014232e-01],
    [9.60949221e-01, 4.18323245e-01, 3.59629789e-01],
    [9.63310281e-01, 4.25389724e-01, 3.59469020e-01],
    [9.65549351e-01, 4.32518707e-01, 3.59529151e-01],
    [9.67671128e-01, 4.39702976e-01, 3.59810172e-01],
    [9.69680441e-01, 4.46935635e-01, 3.60311120e-01],
    [9.71582181e-01, 4.54210170e-01, 3.61030156e-01],
    [9.73381238e-01, 4.61520484e-01, 3.61964652e-01],
    [9.75082439e-01, 4.68860936e-01, 3.63111292e-01],
    [9.76690494e-01, 4.76226350e-01, 3.64466162e-01],
    [9.78209957e-01, 4.83612031e-01, 3.66024854e-01],
    [9.79645181e-01, 4.91013764e-01, 3.67782559e-01],
    [9.81000291e-01, 4.98427800e-01, 3.69734157e-01],
    [9.82279159e-01, 5.05850848e-01, 3.71874301e-01],
    [9.83485387e-01, 5.13280054e-01, 3.74197501e-01],
    [9.84622298e-01, 5.20712972e-01, 3.76698186e-01],
    [9.85692925e-01, 5.28147545e-01, 3.79370774e-01],
    [9.86700017e-01, 5.35582070e-01, 3.82209724e-01],
    [9.87646038e-01, 5.43015173e-01, 3.85209578e-01],
    [9.88533173e-01, 5.50445778e-01, 3.88365009e-01],
    [9.89363341e-01, 5.57873075e-01, 3.91670846e-01],
    [9.90138201e-01, 5.65296495e-01, 3.95122099e-01],
    [9.90871208e-01, 5.72706259e-01, 3.98713971e-01],
    [9.91558165e-01, 5.80106828e-01, 4.02441058e-01],
    [9.92195728e-01, 5.87501706e-01, 4.06298792e-01],
    [9.92784669e-01, 5.94891088e-01, 4.10282976e-01],
    [9.93325561e-01, 6.02275297e-01, 4.14389658e-01],
    [9.93834412e-01, 6.09643540e-01, 4.18613221e-01],
    [9.94308514e-01, 6.16998953e-01, 4.22949672e-01],
    [9.94737698e-01, 6.24349657e-01, 4.27396771e-01],
    [9.95121854e-01, 6.31696376e-01, 4.31951492e-01],
    [9.95480469e-01, 6.39026596e-01, 4.36607159e-01],
    [9.95809924e-01, 6.46343897e-01, 4.41360951e-01],
    [9.96095703e-01, 6.53658756e-01, 4.46213021e-01],
    [9.96341406e-01, 6.60969379e-01, 4.51160201e-01],
    [9.96579803e-01, 6.68255621e-01, 4.56191814e-01],
    [9.96774784e-01, 6.75541484e-01, 4.61314158e-01],
    [9.96925427e-01, 6.82827953e-01, 4.66525689e-01],
    [9.97077185e-01, 6.90087897e-01, 4.71811461e-01],
    [9.97186253e-01, 6.97348991e-01, 4.77181727e-01],
    [9.97253982e-01, 7.04610791e-01, 4.82634651e-01],
    [9.97325180e-01, 7.11847714e-01, 4.88154375e-01],
    [9.97350983e-01, 7.19089119e-01, 4.93754665e-01],
    [9.97350583e-01, 7.26324415e-01, 4.99427972e-01],
    [9.97341259e-01, 7.33544671e-01, 5.05166839e-01],
    [9.97284689e-01, 7.40771893e-01, 5.10983331e-01],
    [9.97228367e-01, 7.47980563e-01, 5.16859378e-01],
    [9.97138480e-01, 7.55189852e-01, 5.22805996e-01],
    [9.97019342e-01, 7.62397883e-01, 5.28820775e-01],
    [9.96898254e-01, 7.69590975e-01, 5.34892341e-01],
    [9.96726862e-01, 7.76794860e-01, 5.41038571e-01],
    [9.96570645e-01, 7.83976508e-01, 5.47232992e-01],
    [9.96369065e-01, 7.91167346e-01, 5.53498939e-01],
    [9.96162309e-01, 7.98347709e-01, 5.59819643e-01],
    [9.95932448e-01, 8.05527126e-01, 5.66201824e-01],
    [9.95680107e-01, 8.12705773e-01, 5.72644795e-01],
    [9.95423973e-01, 8.19875302e-01, 5.79140130e-01],
    [9.95131288e-01, 8.27051773e-01, 5.85701463e-01],
    [9.94851089e-01, 8.34212826e-01, 5.92307093e-01],
    [9.94523666e-01, 8.41386618e-01, 5.98982818e-01],
    [9.94221900e-01, 8.48540474e-01, 6.05695903e-01],
    [9.93865767e-01, 8.55711038e-01, 6.12481798e-01],
    [9.93545285e-01, 8.62858846e-01, 6.19299300e-01],
    [9.93169558e-01, 8.70024467e-01, 6.26189463e-01],
    [9.92830963e-01, 8.77168404e-01, 6.33109148e-01],
    [9.92439881e-01, 8.84329694e-01, 6.40099465e-01],
    [9.92089454e-01, 8.91469549e-01, 6.47116021e-01],
    [9.91687744e-01, 8.98627050e-01, 6.54201544e-01],
    [9.91331929e-01, 9.05762748e-01, 6.61308839e-01],
    [9.90929685e-01, 9.12915010e-01, 6.68481201e-01],
    [9.90569914e-01, 9.20048699e-01, 6.75674592e-01],
    [9.90174637e-01, 9.27195612e-01, 6.82925602e-01],
    [9.89814839e-01, 9.34328540e-01, 6.90198194e-01],
    [9.89433736e-01, 9.41470354e-01, 6.97518628e-01],
    [9.89077438e-01, 9.48604077e-01, 7.04862519e-01],
    [9.88717064e-01, 9.55741520e-01, 7.12242232e-01],
    [9.88367028e-01, 9.62878026e-01, 7.19648627e-01],
    [9.88032885e-01, 9.70012413e-01, 7.27076773e-01],
    [9.87690702e-01, 9.77154231e-01, 7.34536205e-01],
    [9.87386827e-01, 9.84287561e-01, 7.42001547e-01],
    [9.87052509e-01, 9.91437853e-01, 7.49504188e-01],
];

// Parula colormap: https://github.com/BIDS/colormap/blob/master/parula.py
pub const PARULA: [[f32; 3]; 64] = [
    [0.2081, 0.1663, 0.5292],
    [0.2116238095, 0.1897809524, 0.5776761905],
    [0.212252381, 0.2137714286, 0.6269714286],
    [0.2081, 0.2386, 0.6770857143],
    [0.1959047619, 0.2644571429, 0.7279],
    [0.1707285714, 0.2919380952, 0.779247619],
    [0.1252714286, 0.3242428571, 0.8302714286],
    [0.0591333333, 0.3598333333, 0.8683333333],
    [0.0116952381, 0.3875095238, 0.8819571429],
    [0.0059571429, 0.4086142857, 0.8828428571],
    [0.0165142857, 0.4266, 0.8786333333],
    [0.032852381, 0.4430428571, 0.8719571429],
    [0.0498142857, 0.4585714286, 0.8640571429],
    [0.0629333333, 0.4736904762, 0.8554380952],
    [0.0722666667, 0.4886666667, 0.8467],
    [0.0779428571, 0.5039857143, 0.8383714286],
    [0.079347619, 0.5200238095, 0.8311809524],
    [0.0749428571, 0.5375428571, 0.8262714286],
    [0.0640571429, 0.5569857143, 0.8239571429],
    [0.0487714286, 0.5772238095, 0.8228285714],
    [0.0343428571, 0.5965809524, 0.819852381],
    [0.0265, 0.6137, 0.8135],
    [0.0238904762, 0.6286619048, 0.8037619048],
    [0.0230904762, 0.6417857143, 0.7912666667],
    [0.0227714286, 0.6534857143, 0.7767571429],
    [0.0266619048, 0.6641952381, 0.7607190476],
    [0.0383714286, 0.6742714286, 0.743552381],
    [0.0589714286, 0.6837571429, 0.7253857143],
    [0.0843, 0.6928333333, 0.7061666667],
    [0.1132952381, 0.7015, 0.6858571429],
    [0.1452714286, 0.7097571429, 0.6646285714],
    [0.1801333333, 0.7176571429, 0.6424333333],
    [0.2178285714, 0.7250428571, 0.6192619048],
    [0.2586428571, 0.7317142857, 0.5954285714],
    [0.3021714286, 0.7376047619, 0.5711857143],
    [0.3481666667, 0.7424333333, 0.5472666667],
    [0.3952571429, 0.7459, 0.5244428571],
    [0.4420095238, 0.7480809524, 0.5033142857],
    [0.4871238095, 0.7490619048, 0.4839761905],
    [0.5300285714, 0.7491142857, 0.4661142857],
    [0.5708571429, 0.7485190476, 0.4493904762],
    [0.609852381, 0.7473142857, 0.4336857143],
    [0.6473, 0.7456, 0.4188],
    [0.6834190476, 0.7434761905, 0.4044333333],
    [0.7184095238, 0.7411333333, 0.3904761905],
    [0.7524857143, 0.7384, 0.3768142857],
    [0.7858428571, 0.7355666667, 0.3632714286],
    [0.8185047619, 0.7327333333, 0.3497904762],
    [0.8506571429, 0.7299, 0.3360285714],
    [0.8824333333, 0.7274333333, 0.3217],
    [0.9139333333, 0.7257857143, 0.3062761905],
    [0.9449571429, 0.7261142857, 0.2886428571],
    [0.9738952381, 0.7313952381, 0.266647619],
    [0.9937714286, 0.7454571429, 0.240347619],
    [0.9990428571, 0.7653142857, 0.2164142857],
    [0.9955333333, 0.7860571429, 0.196652381],
    [0.988, 0.8066, 0.1793666667],
    [0.9788571429, 0.8271428571, 0.1633142857],
    [0.9697, 0.8481380952, 0.147452381],
    [0.9625857143, 0.8705142857, 0.1309],
    [0.9588714286, 0.8949, 0.1132428571],
    [0.9598238095, 0.9218333333, 0.0948380952],
    [0.9661, 0.9514428571, 0.0755333333],
    [0.9763, 0.9831, 0.0538],
];

// CB dark red: rgb(226, 26, 27) ranged from 0.0 to 1.0
pub const COLOR_PRIMARY_X: Color = Color::rgb(0.8862745098, 0.10196078431, 0.10588235294);
// CB dark blue: rgb(30, 119, 179) ranged from 0.0 to 1.0
pub const COLOR_PRIMARY_Y: Color = Color::rgb(0.11764705882, 0.46666666666, 0.70196078431);
// CB yellow: rgb(255, 215, 0) ranged from 0.0 to 1.0
pub const COLOR_PRIMARY_Z: Color = Color::rgb(1., 1., 0.6);

// CB light red: rgb(250, 153, 153) ranged from 0.0 to 1.0
pub const COLOR_PRIMARY_X_LIGHT: Color = Color::rgb(0.98039215686, 0.6, 0.6);
// CB light blue: rgb(166, 205, 226) ranged from 0.0 to 1.0
pub const COLOR_PRIMARY_Y_LIGHT: Color = Color::rgb(0.65098039215, 0.80392156862, 0.8862745098);
// white: rgb(255, 255, 255) ranged from 0.0 to 1.0
pub const COLOR_PRIMARY_Z_LIGHT: Color = Color::rgb(1.0, 1.0, 0.8);

pub fn color_map(value: f32, colors: Vec<[f32; 3]>) -> Color {
    let index = std::cmp::min(
        colors.len() - 1,
        (value * (colors.len() - 1) as f32).round() as usize,
    );
    Color::rgb(colors[index][0], colors[index][1], colors[index][2])
}

// Construct a mesh object that can be rendered using the Bevy framework.
pub fn get_bevy_mesh_of_regions(
    regions: &Vec<Surface>,
    patch_graph: &Doconeli,
    granulated_mesh: &Doconeli,
    color_type: ColorType,
    configuration: &Configuration,
    evaluation: &Evaluation,
) -> Mesh {
    let mut mesh_triangle_list = Mesh::new(PrimitiveTopology::TriangleList);
    let mut vertex_positions = vec![];
    let mut vertex_normals = vec![];
    let mut vertex_colors = vec![];

    for surface in regions {
        let color = match color_type {
            ColorType::DirectionPrimary => {
                get_color(surface.direction.unwrap(), true, configuration)
            }
            ColorType::DirectionSecondary => surface.color.unwrap().into(),
            ColorType::Random => get_random_color(),
            ColorType::Static(color) => color,
            _ => Color::PINK,
        };

        let mut color_f32 = color.as_rgba_f32();

        for (i, subface) in surface.faces.iter().enumerate() {
            if color_type == ColorType::DistortionAlignment {
                color_f32 = color_map(
                    evaluation.face_to_fidelity[&subface.face_id],
                    MAGMA.to_vec(),
                )
                .as_rgba_f32();
            }
            if color_type == ColorType::DistortionJacobian {
                color_f32 =
                    color_map(evaluation.patch_to_angle[&surface.id], MAGMA.to_vec()).as_rgba_f32();
            }
            for &p1 in &subface.bounding_points {
                for &p2 in &subface.bounding_points {
                    for &p3 in &subface.bounding_points {
                        vertex_positions.push(p1.0);
                        vertex_positions.push(p2.0);
                        vertex_positions.push(p3.0);
                        vertex_normals.push(p1.1);
                        vertex_normals.push(p2.1);
                        vertex_normals.push(p3.1);
                        vertex_colors.push(color_f32);
                        vertex_colors.push(color_f32);
                        vertex_colors.push(color_f32);
                    }
                }
            }
        }
    }

    let length = vertex_positions.len();
    mesh_triangle_list.insert_attribute(Mesh::ATTRIBUTE_POSITION, vertex_positions);
    mesh_triangle_list.insert_attribute(Mesh::ATTRIBUTE_NORMAL, vertex_normals);
    mesh_triangle_list.insert_attribute(Mesh::ATTRIBUTE_COLOR, vertex_colors);
    mesh_triangle_list.set_indices(Some(Indices::U32((0..length as u32).collect())));

    mesh_triangle_list
}

pub fn get_labeling_of_mesh(
    path: &PathBuf,
    granulated_mesh: &Doconeli,
    regions: &Vec<Surface>,
) -> Result<(), Box<dyn Error>> {
    let mut labels = vec![-1; granulated_mesh.faces.len()];

    for surface in regions {
        // get avg normal of surface
        let avg_normal = average(
            surface
                .faces
                .iter()
                .map(|subface| granulated_mesh.faces[subface.face_id].normal),
        );
        // positive or negative based on angle with the direction
        let positive = surface.direction.unwrap().to_vector().dot(avg_normal) > 0.;
        // set label for all faces in the surface
        let label = match surface.direction.unwrap() {
            PrincipalDirection::X => {
                if positive {
                    0
                } else {
                    1
                }
            }
            PrincipalDirection::Y => {
                if positive {
                    2
                } else {
                    3
                }
            }
            PrincipalDirection::Z => {
                if positive {
                    4
                } else {
                    5
                }
            }
        };

        for subface in surface.faces.iter() {
            labels[subface.face_id] = label;
        }
    }

    let mut file = std::fs::File::create(path)?;

    for label in labels {
        writeln!(file, "{label:?}");
    }

    Ok(())
}

// Construct a mesh object that can be rendered using the Bevy framework.
pub fn get_bevy_mesh_of_mesh(
    mesh: &Doconeli,
    color_type: ColorType,
    configuration: &Configuration,
) -> Mesh {
    let mut mesh_triangle_list = Mesh::new(PrimitiveTopology::TriangleList);
    let mut vertex_positions = Vec::with_capacity(mesh.faces.len() * 3);
    let mut vertex_normals = Vec::with_capacity(mesh.faces.len() * 3);
    let mut vertex_colors = Vec::with_capacity(mesh.faces.len() * 3);

    for face_id in 0..mesh.faces.len() {
        let color = match color_type {
            ColorType::Static(c) => c,
            ColorType::Labeling => match mesh.faces[face_id].label {
                Some(0) => COLOR_PRIMARY_X,
                Some(1) => COLOR_PRIMARY_X_LIGHT,
                Some(2) => COLOR_PRIMARY_Y,
                Some(3) => COLOR_PRIMARY_Y_LIGHT,
                Some(4) => COLOR_PRIMARY_Z,
                Some(5) => COLOR_PRIMARY_Z_LIGHT,
                _ => Color::WHITE,
            },
            _ => mesh.faces[face_id].color,
        };

        for vertex_id in mesh.get_vertices_of_face(face_id) {
            vertex_positions.push(mesh.get_position_of_vertex(vertex_id));
            vertex_normals.push(mesh.vertices[vertex_id].normal);
            vertex_colors.push(color.as_rgba_f32());
        }
    }

    mesh_triangle_list.insert_attribute(Mesh::ATTRIBUTE_POSITION, vertex_positions);
    mesh_triangle_list.insert_attribute(Mesh::ATTRIBUTE_NORMAL, vertex_normals);
    mesh_triangle_list.insert_attribute(Mesh::ATTRIBUTE_COLOR, vertex_colors);
    mesh_triangle_list.set_indices(Some(Indices::U32(
        (0..mesh.faces.len() as u32 * 3).collect(),
    )));

    mesh_triangle_list
}

// Construct a mesh object that can be rendered using the Bevy framework.
pub fn get_bevy_mesh_of_graph(
    graph: &Doconeli,
    color_type: ColorType,
    configuration: &Configuration,
) -> Mesh {
    let mut mesh_triangle_list = Mesh::new(PrimitiveTopology::TriangleList);
    let mut vertex_positions = vec![];
    let mut vertex_normals = vec![];
    let mut vertex_colors = vec![];

    for face_id in 0..graph.faces.len() {
        let vertices = graph.get_vertices_of_face(face_id);
        let edges = graph.get_edges_of_face(face_id);

        let dir = [
            PrincipalDirection::X,
            PrincipalDirection::Y,
            PrincipalDirection::Z,
        ]
        .into_iter()
        .filter(|&dir| {
            !edges
                .iter()
                .map(|&edge_id| graph.edges[edge_id].direction)
                .contains(&Some(dir))
        })
        .next();

        let color = match color_type {
            ColorType::Static(color) => color,
            ColorType::Random => get_random_color(),
            ColorType::DirectionPrimary => get_color(dir.unwrap(), true, configuration),
            ColorType::DirectionSecondary => get_color(dir.unwrap(), false, configuration),
            _ => todo!(),
        };

        for &vertex1 in &vertices {
            for &vertex2 in &vertices {
                for &vertex3 in &vertices {
                    vertex_positions.push(graph.get_position_of_vertex(vertex1));
                    vertex_normals.push(graph.vertices[vertex1].normal);
                    vertex_colors.push(color.as_rgba_f32());
                    vertex_positions.push(graph.get_position_of_vertex(vertex2));
                    vertex_normals.push(graph.vertices[vertex2].normal);
                    vertex_colors.push(color.as_rgba_f32());
                    vertex_positions.push(graph.get_position_of_vertex(vertex3));
                    vertex_normals.push(graph.vertices[vertex3].normal);
                    vertex_colors.push(color.as_rgba_f32());
                }
            }
        }
    }

    mesh_triangle_list.set_indices(Some(Indices::U32(
        (0..vertex_positions.len() as u32).collect(),
    )));
    mesh_triangle_list.insert_attribute(Mesh::ATTRIBUTE_POSITION, vertex_positions);
    mesh_triangle_list.insert_attribute(Mesh::ATTRIBUTE_NORMAL, vertex_normals);
    mesh_triangle_list.insert_attribute(Mesh::ATTRIBUTE_COLOR, vertex_colors);

    mesh_triangle_list
}

pub fn intersection_exact_in_2d(p1: Vec2, p2: Vec2, p3: Vec2, p4: Vec2) -> Option<Vec2> {
    // https://en.wikipedia.org/wiki/Line%E2%80%93line_intersection
    let t = ((p1.x - p3.x) * (p3.y - p4.y) - (p1.y - p3.y) * (p3.x - p4.x))
        / ((p1.x - p2.x) * (p3.y - p4.y) - (p1.y - p2.y) * (p3.x - p4.x));

    let u = ((p1.x - p3.x) * (p1.y - p2.y) - (p1.y - p3.y) * (p1.x - p2.x))
        / ((p1.x - p2.x) * (p3.y - p4.y) - (p1.y - p2.y) * (p3.x - p4.x));

    if t >= 0. && t <= 1. && u >= 0. && u <= 1. {
        let intersection_x = p1.x + t * (p2.x - p1.x);
        let intersection_y = p1.y + t * (p2.y - p1.y);

        return Some(Vec2::new(intersection_x, intersection_y));
    }

    None
}

pub fn convert_3d_to_2d(point: Vec3, reference: Vec3) -> Vec2 {
    let alpha = point.angle_between(reference);
    Vec2::new(point.length() * alpha.cos(), point.length() * alpha.sin())
}

pub fn point_lies_in_triangle(point: Vec3, triangle: (Vec3, Vec3, Vec3)) -> bool {
    let vectors = vec![triangle.0 - point, triangle.1 - point, triangle.2 - point];

    let sum_of_angles = vectors[0].angle_between(vectors[1])
        + vectors[1].angle_between(vectors[2])
        + vectors[2].angle_between(vectors[0]);

    return (sum_of_angles - 2. * PI).abs() < 0.0001;
}

pub fn point_lies_in_segment(point: Vec3, segment: (Vec3, Vec3)) -> bool {
    let segment_length = segment.0.distance(segment.1);

    let segment_length_through_point = point.distance(segment.0) + point.distance(segment.1);

    return (segment_length - segment_length_through_point).abs() < 0.00001;
}
