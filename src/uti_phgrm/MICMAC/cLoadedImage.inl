template <class TypeEl,class tBase> cInterpolateurIm2D<TypeEl>  * InterpoleOfEtape(const cEtapeMEC & anEt,TypeEl *,tBase *)
// template <class TypeEl,class tBase> cInterpolateurIm2D<TypeEl>  InterpoleOfEtape(const cEtapeMEC & anEt)
{
   double aCoef3 = anEt.CoefInterpolationBicubique().ValWithDef(-0.5);
   switch(anEt.ModeInterpolation().Val())
   {
       case eInterpolPPV :
         return new cInterpolPPV<TypeEl>;
       break;

       case eInterpolBiLin :
// std::cout << "BIIIIIIIIIIIIILLLINNNNN \n";
         return  new cInterpolBilineaire<TypeEl>;
       break;
       case eInterpolBiCub :
         return new cInterpolBicubique<TypeEl>(aCoef3);
       break;

       case eInterpolBicubOpt :
         return new cTplCIKTabul<TypeEl,tBase>(10,8,aCoef3);
       break;

       case eInterpolMPD :
// std::cout << "MPDDDDDDDDDDDDDDD \n";
         return  new cTplCIKTabul<TypeEl,tBase>(10,8,0.0,eTabulMPD_EcartMoyen);
       break;

       case eInterpolSinCard :
       {
           double aSzK = anEt.SzSinCard().ValWithDef(5.0);
           double aSzA = anEt.SzAppodSinCard().ValWithDef(5.0);
           int aNbD = anEt.NdDiscKerInterp().ValWithDef(1000);

          cSinCardApodInterpol1D aKer(cSinCardApodInterpol1D::eTukeyApod,aSzK,aSzA,1e-4,false);

          return  new cTabIM2D_FromIm2D<TypeEl>(&aKer,aNbD,false);
          break;
      }
      case eOldInterpolSinCard :
      {
         return  new cInterpolSinusCardinal<TypeEl>(anEt.TailleFenetreSinusCardinal().Val(), anEt.ApodisationSinusCardinal().Val());
         break;
      }

   }
   ELISE_ASSERT(false,"Incoh in InterpoleOfEtape");
   return 0;

}
