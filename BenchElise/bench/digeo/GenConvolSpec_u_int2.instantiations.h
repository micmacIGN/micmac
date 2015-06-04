template <> void cConvolSpec<U_INT2>::init()
{
	static bool theFirst = true;
	if ( !theFirst ) return;
	theFirst = false;
	{
		INT theCoeff[13] = {9,71,390,1465,3774,6655,8040,6655,3774,1465,390,71,9};
		new cConvolSpec_U_INT2_Num0(theCoeff);
	}
}
