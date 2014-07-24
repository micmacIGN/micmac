template <> inline void cAppliDigeo::__InitConvolSpec<U_INT2>()
{
	static bool theFirst = true;
	if ( !theFirst ) return;
	theFirst = false;
	{
		INT theCoeff[7] = {23,883,7662,15632,7662,883,23};
		new cConvolSpec_U_INT2_Num0(theCoeff);
	}
	{
		INT theCoeff[9] = {5,162,1852,7933,12864,7933,1852,162,5};
		new cConvolSpec_U_INT2_Num1(theCoeff);
	}
	{
		INT theCoeff[11] = {4,67,609,2945,7573,10372,7573,2945,609,67,4};
		new cConvolSpec_U_INT2_Num2(theCoeff);
	}
	{
		INT theCoeff[13] = {6,53,326,1346,3702,6793,8316,6793,3702,1346,326,53,6};
		new cConvolSpec_U_INT2_Num3(theCoeff);
	}
	{
		INT theCoeff[17] = {2,12,64,263,842,2078,3964,5838,6642,5838,3964,2078,842,263,64,12,2};
		new cConvolSpec_U_INT2_Num4(theCoeff);
	}
	{
		INT theCoeff[21] = {1,7,28,95,276,682,1426,2531,3814,4877,5294,4877,3814,2531,1426,682,276,95,28,7,1};
		new cConvolSpec_U_INT2_Num5(theCoeff);
	}
}
