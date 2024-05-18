use arrayfire;





const ZERO_F64: f64 = 0.0;
const HIGH_F64: f64 = 1000000.0;
const TWO_F64: f64 = 2.0;


pub fn ReLU<Z: arrayfire::RealFloating>(X: &arrayfire::Array<Z>) -> arrayfire::Array<Z>
{
    let single_dims = arrayfire::Dim4::new(&[1,1,1,1]);
    let ZERO = arrayfire::constant::<f64>(ZERO_F64,single_dims).cast::<Z>();
    let HIGH = arrayfire::constant::<f64>(HIGH_F64,single_dims).cast::<Z>();


	arrayfire::clamp(X, &ZERO, &HIGH, false)
}


pub fn Softplus<Z: arrayfire::RealFloating<UnaryOutType = Z,AbsOutType = Z>  >(X: &arrayfire::Array<Z>) -> arrayfire::Array<Z>
{
    let single_dims = arrayfire::Dim4::new(&[1,1,1,1]);
    let ZERO = arrayfire::constant::<f64>(ZERO_F64,single_dims).cast::<Z>();

	let mut temp = ZERO-arrayfire::abs(X);
	temp = arrayfire::exp(&temp);
	ReLU(X)  +  arrayfire::log1p(&temp)
}





pub fn UAF<Z: arrayfire::RealFloating  >(
	X: &arrayfire::Array<Z>,
	A: &arrayfire::Array<Z>,
	B: &arrayfire::Array<Z>,
	C: &arrayfire::Array<Z>,
	D: &arrayfire::Array<Z>,
	E: &arrayfire::Array<Z>) ->  arrayfire::Array<Z> {


	// X + B
	let mut temp0 = arrayfire::add(X, B, true);
	// X^2
	let mut temp1 = arrayfire::pow(X,&TWO_F64,false);

	// -|C|
	let mut temp2 = -arrayfire::abs(C);

	//A(X + B)  +  -|C|( X^2 )
	temp0 = arrayfire::mul(A,&temp0,true) + arrayfire::mul(&temp2,&temp1,true);

	drop(temp2);

	// X - B;
	temp1 = arrayfire::sub(X, B, true);
	// D (X - B)
	temp1 = arrayfire::mul(D,&temp1,true);


	//Softplus( A(X + B)  +  C( X^2 ) )
	temp0 = Softplus(&temp0);
	//Softplus( D (X - B) )
	temp1 = Softplus(&temp1);

	//  Softplus( A(X + B)  +  C( X^2 ) ) - Softplus( D (X - B) )
	temp0 = temp0 - temp1;

	// Softplus( A(X + B)  +  C( X^2 ) ) - Softplus( D (X - B) ) + E
	arrayfire::add(&temp0, E, true)
}



















pub fn deriUAF(
	X: &arrayfire::Array<f64>,
	A: &arrayfire::Array<f64>,
	B: &arrayfire::Array<f64>,
	C: &arrayfire::Array<f64>,
	D: &arrayfire::Array<f64>,
	E: &arrayfire::Array<f64>,
	dX: &mut arrayfire::Array<f64>,
	dA: &mut arrayfire::Array<f64>,
	dB: &mut arrayfire::Array<f64>,
	dC: &mut arrayfire::Array<f64>,
	dD: &mut arrayfire::Array<f64>,
	dE: &mut arrayfire::Array<f64>)
{

	// X + B 
	let mut temp0 = arrayfire::add(X, B, true);
	// X^2
	let mut temp1 = arrayfire::pow(X,&TWO_F64,false);

	// -|C|
	let mut temp2 = -arrayfire::abs(C);


	//A(X + B)   -  |C|( X^2 )
	let mut expcal0 = arrayfire::mul(A,&temp0,true) + arrayfire::mul(&temp2,&temp1,true);

	//Sigmoid( A(X + B)   -  |C|( X^2 ) )
	expcal0 = arrayfire::sigmoid(&expcal0);

	//dA = (X + B) Sigmoid( A(X + B)   -  |C|( X^2 ) )
	*dA = arrayfire::mul(&temp0,&expcal0,false);
	//dB = (A) Sigmoid( A(X + B)   -  |C|( X^2 ) )
	*dB = arrayfire::mul(A,&expcal0,true);

	// -sign(C)
	temp2 = 2.0f64*(arrayfire::sign(C) - 0.5f64);

	// -sign(C) (X^2)
	temp1 = arrayfire::mul(&temp2, &temp1, true);

	//dC = -sign(C)(X^2) Sigmoid( A(X + B)  +  -|C|( X^2 ) )
	*dC = arrayfire::mul(&temp1,&expcal0,false);




	// -|C|
	temp2 = -arrayfire::abs(C);


	//A - 2|C|x
	temp0 = TWO_F64*arrayfire::mul(&temp2,X,true);
	temp0 = arrayfire::add(A, &temp0, true);
	// (A - 2|C|x) Sigmoid( A(X + B)  -  |C|( X^2 ) )
	expcal0 = arrayfire::mul(&temp0,&expcal0,false);

	// X - B
	temp0 = arrayfire::sub(X, B, true);
	//D ( X - B )
	let mut expcal1 = arrayfire::mul(D,&temp0,true);

	//Sigmoid( D ( X - B )  )
	expcal1 = arrayfire::sigmoid(&expcal1);

	*dB = dB.clone() + arrayfire::mul(D,&expcal1 ,true);

	//dD = - (X - B) Sigmoid( D ( X - B )  )
	*dD = -arrayfire::mul(&temp0,&expcal1,false);
	//dE = 1
	*dE = arrayfire::constant::<f64>(1.0, X.dims());

	expcal1 = arrayfire::mul(D,&expcal1,true);

	//dX = (A - 2|C|x) Sigmoid( A(X + B)  - |C|( X^2 ) ) - D Sigmoid( D ( X - B )  )
	*dX = (expcal0-expcal1);
}








