int CPP_Yann_main(int argc, char** argv)
{

	std::string aIn;
	std::string aFile2D;
	std::string aFile3D;
	bool test;
	bool ApplyD=false;
	bool Save=false;

	 ElInitArgMain
    	(
        	argc, argv,
        	LArgMain() << EAMC(aIn,"Dir of calibration")
		           << EAMC(aFile2D,"Im measure files")
		           << EAMC(aFile3D,"Terr measure files")
		           << EAMC(test,"boolean to test"),
        	LArgMain() << EAM (ApplyD,"Dist",true,"Apply distortion?")
		           << EAM (Save,"Save",true,"Save?")
    	);

#if (ELISE_windows)
      replace( aIn.begin(), aIn.end(), '\\', '/' );
#endif


	std::cout << "voici ma fonction " << aIn << " " << ApplyD << "\n";

	return 1;
}
