
#include "GpGpu/GpGpu_InterOptimisation.h"
#include "GpGpu/SData2Optimize.h"

extern "C" void TestOptimisationOneDirectionZ(Data2Optimiz<CuDeviceData3D> &d2O);

int main()
{
    cout << "Launch Data optimisation GpGpu ***" << endl;

    // Declaration des variables du cote du HOST
    HOST_Data2Opti h2O;
    // Declaration des variables du cote du DEVICE
    DEVC_Data2Opti d2O;

    uint nbLines    = 100;
    uint lLines     = 500;
    uint depth      = NAPPEMAX;

    short2 dZ   = make_short2(-depth/2,depth/2);
    depth       = count(dZ);

    uint sizeMaxLine = (uint)(1.5f*sqrt((float)lLines * lLines + nbLines * nbLines));

    h2O.ReallocParam(sizeMaxLine);
    d2O.ReallocParam(sizeMaxLine);

    uint pit_Strm_DZ    = WARPSIZE;
    uint pit_Strm_ICost = NAPPEMAX;

    CuHostData3D<ushort> tabZ(nbLines,lLines,2);

    tabZ.FillRandom(0,depth/2);

    for (uint p= 0 ; p < nbLines; p++)
    {

        h2O.SetParamLine(p,pit_Strm_ICost,pit_Strm_DZ,lLines);

        uint sizeStreamLine = 0; // lLines * depth;
        for (uint aK= 0 ; aK < lLines; aK++)      
            sizeStreamLine += count(make_short2(-tabZ[make_uint3(p,aK,0)],tabZ[make_uint3(p,aK,1)]));

        pit_Strm_DZ     += iDivUp(lLines,       WARPSIZE) * WARPSIZE;
        pit_Strm_ICost  += iDivUp(sizeStreamLine,   WARPSIZE) * WARPSIZE;
    }

    h2O.ReallocInputIf(pit_Strm_ICost + NAPPEMAX,pit_Strm_DZ + WARPSIZE);
    h2O._s_InitCostVol.Fill(0);

    // index
    for (uint idLine= 0 ; idLine < nbLines; idLine++)
    {
        uint    pitStrm = 0;

        for (uint aK= 0 ; aK < lLines; aK++)

        {
            short2 lDZ      = make_short2(-tabZ[make_uint3(idLine,aK,0)],tabZ[make_uint3(idLine,aK,1)]);
            ushort lDepth   = count(lDZ);

            h2O._s_Index[h2O._param[0][idLine].y + aK ] = lDZ;

            uint idStrm = h2O._param[0][idLine].x + pitStrm - lDZ.x;

            for ( int aPx = lDZ.x ; aPx < lDZ.y; aPx++)
                h2O._s_InitCostVol[idStrm + aPx]  = 10000 * (idLine + 1) + (aK+1) * 1000 + aPx - lDZ.x + 1;

            pitStrm += lDepth;
        }
    }

    h2O.SetNbLine(nbLines);    
    d2O.SetNbLine(h2O._nbLines);

    h2O.ReallocOutputIf(h2O._s_InitCostVol.GetSize());

    h2O._s_ForceCostVol[0].Fill(0);

    d2O.ReallocIf(h2O);

    //      Transfert des données vers le device                            ---------------		-
    d2O.CopyHostToDevice(h2O);
    d2O._s_ForceCostVol[0].CopyHostToDevice(h2O._s_ForceCostVol[0].pData());

    //h2O._s_InitCostVol.OutputValues();

    TestOptimisationOneDirectionZ(d2O);

    d2O.CopyDevicetoHost(h2O);

    //h2O._s_ForceCostVol[0].OutputValues();

    //
    uint errorCount = 0;

    for (uint idLine= 0 ; idLine < nbLines; idLine++)
    {
        uint    pitStrm = 0;

        for (uint aK= 0 ; aK < lLines; aK++)
        {
            short2 dZ = h2O._s_Index[h2O._param[0][idLine].y + aK ];

            uint idStrm = h2O._param[0][idLine].x + pitStrm - dZ.x;

            for ( int aPx = dZ.x ; aPx < dZ.y; aPx++)
                if( h2O._s_InitCostVol[idStrm + aPx]  != h2O._s_ForceCostVol[0][idStrm + aPx])
                {
                    printf(" %d ",h2O._s_InitCostVol[idStrm + aPx]);
                    errorCount++;
                }

            pitStrm += count(dZ);
        }
    }

    printf("\nError Count   = %d/%d\n",errorCount,h2O._s_InitCostVol.GetSize()- 2*NAPPEMAX);
    printf("Error percent = %f\n",(((float)errorCount*100)/(h2O._s_InitCostVol.GetSize()- 2*NAPPEMAX)));

    return 0;
}

