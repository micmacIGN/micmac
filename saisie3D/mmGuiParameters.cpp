#include "mmGuiParameters.h"

//Qt
#include <QSettings>

#include "util.h"

//System
#include <string.h>

//! Static unique instance of mmGui
static mmGui* s_gui = 0;

const int c_fColorArraySize = 4*sizeof(float);
const int c_ubColorArraySize = 3*sizeof(unsigned char);

const mmGui::ParamStruct& mmGui::Parameters()
{
    if (!s_gui)
    {
        s_gui = new mmGui();
        s_gui->params.fromPersistentSettings();
    }

    return s_gui->params;
}

void mmGui::ReleaseInstance()
{
    if (s_gui)
        delete s_gui;
    s_gui=0;
}

void mmGui::Set(const ParamStruct& params)
{
    if (!s_gui)
        s_gui = new mmGui();

    s_gui->params = params;
}

mmGui::ParamStruct::ParamStruct()
{
    reset();
}

void mmGui::ParamStruct::reset()
{
    memcpy(lightAmbientColor,   mmColor::night,				    c_fColorArraySize);
    memcpy(lightSpecularColor,  mmColor::darker,				c_fColorArraySize);
    memcpy(lightDiffuseColor,   mmColor::lighter,				c_fColorArraySize);
    memcpy(meshSpecular,        mmColor::middle,				c_fColorArraySize);
    memcpy(pointsDefaultCol,    mmColor::defaultColor,			c_ubColorArraySize);
    memcpy(textDefaultCol,      mmColor::defaultColor,			c_ubColorArraySize);
    memcpy(backgroundCol,       mmColor::defaultBkgColor,		c_ubColorArraySize);
    memcpy(histBackgroundCol,	mmColor::defaultHistBkgColor,	c_ubColorArraySize);
    memcpy(labelCol,			mmColor::defaultLabelColor,		c_ubColorArraySize);
    memcpy(bbDefaultCol,        mmColor::yellow,				c_ubColorArraySize);

    drawBackgroundGradient      = true;
    decimateMeshOnMove          = true;
    decimateCloudOnMove         = true;
    displayCross                = true;

    colorScaleAlwaysSymmetrical	= true;
    colorScaleAlwaysShowZero	= true;
    colorScaleSquareSize		= 20;

    defaultFontSize				= 10;
    displayedNumPrecision		= 6;
    labelsTransparency			= 50;
}

mmGui::ParamStruct& mmGui::ParamStruct::operator =(const mmGui::ParamStruct& params)
{
    memcpy(lightDiffuseColor,   params.lightDiffuseColor,   c_fColorArraySize);
    memcpy(lightAmbientColor,   params.lightAmbientColor,   c_fColorArraySize);
    memcpy(lightSpecularColor,  params.lightSpecularColor,  c_fColorArraySize);
    memcpy(meshSpecular,		params.meshSpecular,        c_fColorArraySize);
    memcpy(pointsDefaultCol,    params.pointsDefaultCol,    c_ubColorArraySize);
    memcpy(textDefaultCol,      params.textDefaultCol,      c_ubColorArraySize);
    memcpy(backgroundCol,       params.backgroundCol,       c_ubColorArraySize);
    memcpy(histBackgroundCol,	params.histBackgroundCol,	c_ubColorArraySize);
    memcpy(labelCol,			params.labelCol,			c_ubColorArraySize);
    memcpy(bbDefaultCol,        params.bbDefaultCol,        c_ubColorArraySize);

    drawBackgroundGradient      = params.drawBackgroundGradient;
    decimateMeshOnMove          = params.decimateMeshOnMove;
    decimateCloudOnMove         = params.decimateCloudOnMove;
    displayCross                = params.displayCross;
    colorScaleAlwaysSymmetrical	= params.colorScaleAlwaysSymmetrical;
    colorScaleAlwaysShowZero	= params.colorScaleAlwaysShowZero;
    colorScaleSquareSize		= params.colorScaleSquareSize;
    defaultFontSize				= params.defaultFontSize;
    displayedNumPrecision		= params.displayedNumPrecision;
    labelsTransparency			= params.labelsTransparency;

    return *this;

}

void mmGui::ParamStruct::fromPersistentSettings()
{
    QSettings settings;
    settings.beginGroup("OpenGL");

    memcpy(lightAmbientColor,   settings.value("lightAmbientColor",     QByteArray((const char*)mmColor::night,                 c_fColorArraySize)).toByteArray().data(), c_fColorArraySize);
    memcpy(lightSpecularColor,  settings.value("lightSpecularColor",    QByteArray((const char*)mmColor::middle,                c_fColorArraySize)).toByteArray().data(), c_fColorArraySize);
    memcpy(lightDiffuseColor,   settings.value("lightDiffuseColor",     QByteArray((const char*)mmColor::lighter,               c_fColorArraySize)).toByteArray().data(), c_fColorArraySize);
    memcpy(meshSpecular,        settings.value("meshSpecular",          QByteArray((const char*)mmColor::middle,				c_fColorArraySize)).toByteArray().data(), c_fColorArraySize);
    memcpy(pointsDefaultCol,    settings.value("pointsDefaultColor",    QByteArray((const char*)mmColor::defaultColor,          c_ubColorArraySize)).toByteArray().data(), c_ubColorArraySize);
    memcpy(textDefaultCol,      settings.value("textDefaultColor",      QByteArray((const char*)mmColor::defaultColor,          c_ubColorArraySize)).toByteArray().data(), c_ubColorArraySize);
    memcpy(backgroundCol,       settings.value("backgroundColor",       QByteArray((const char*)mmColor::defaultBkgColor,       c_ubColorArraySize)).toByteArray().data(), c_ubColorArraySize);
    memcpy(histBackgroundCol,	settings.value("histBackgroundColor",	QByteArray((const char*)mmColor::defaultHistBkgColor,   c_ubColorArraySize)).toByteArray().data(), c_ubColorArraySize);
    memcpy(labelCol,			settings.value("labelColor",			QByteArray((const char*)mmColor::defaultLabelColor,     c_ubColorArraySize)).toByteArray().data(), c_ubColorArraySize);
    memcpy(bbDefaultCol,        settings.value("bbDefaultColor",        QByteArray((const char*)mmColor::yellow,                c_ubColorArraySize)).toByteArray().data(), c_ubColorArraySize);

    drawBackgroundGradient  = settings.value("backgroundGradient", true).toBool();
    decimateMeshOnMove      = settings.value("meshDecimation", true).toBool();
    decimateCloudOnMove     = settings.value("cloudDecimation", true).toBool();
    displayCross            = settings.value("crossDisplayed", true).toBool();

    colorScaleAlwaysSymmetrical	= settings.value("colorScaleAlwaysSymmetrical", true).toBool();
    colorScaleAlwaysShowZero	= settings.value("colorScaleAlwaysShowZero", true).toBool();
    colorScaleSquareSize		= (unsigned)settings.value("colorScaleSquareSize", 20).toInt();

    defaultFontSize				= (unsigned)settings.value("defaultFontSize", 10).toInt();
    displayedNumPrecision		= (unsigned)settings.value("displayedNumPrecision", 6).toInt();
    labelsTransparency			= (unsigned)settings.value("labelsTransparency", 50).toInt();

    settings.endGroup();
}

void mmGui::ParamStruct::toPersistentSettings()
{
    QSettings settings;
    settings.beginGroup("OpenGL");

    settings.setValue("lightDiffuseColor",QByteArray((const char*)lightDiffuseColor,c_fColorArraySize));
    settings.setValue("lightAmbientColor",QByteArray((const char*)lightAmbientColor,c_fColorArraySize));
    settings.setValue("lightSpecularColor",QByteArray((const char*)lightSpecularColor,c_fColorArraySize));
    settings.setValue("meshFrontDiff",QByteArray((const char*)meshFrontDiff,c_fColorArraySize));
    settings.setValue("pointsDefaultColor",QByteArray((const char*)pointsDefaultCol,c_ubColorArraySize));
    settings.setValue("textDefaultColor",QByteArray((const char*)textDefaultCol,c_ubColorArraySize));
    settings.setValue("backgroundColor",QByteArray((const char*)backgroundCol,c_ubColorArraySize));
    settings.setValue("histBackgroundColor",QByteArray((const char*)histBackgroundCol,c_ubColorArraySize));
    settings.setValue("labelColor",QByteArray((const char*)labelCol,c_ubColorArraySize));
    settings.setValue("bbDefaultColor",QByteArray((const char*)bbDefaultCol,c_ubColorArraySize));
    settings.setValue("backgroundGradient",drawBackgroundGradient);
    settings.setValue("meshDecimation",decimateMeshOnMove);
    settings.setValue("cloudDecimation",decimateCloudOnMove);
    settings.setValue("crossDisplayed",displayCross);
    settings.setValue("colorScaleAlwaysSymmetrical", colorScaleAlwaysSymmetrical);
    settings.setValue("colorScaleAlwaysShowZero", colorScaleAlwaysShowZero);
    settings.setValue("colorScaleSquareSize", colorScaleSquareSize);
    settings.setValue("defaultFontSize", defaultFontSize);
    settings.setValue("displayedNumPrecision", displayedNumPrecision);
    settings.setValue("labelsTransparency", labelsTransparency);


    settings.endGroup();
}
