#include "GLWidgetSet.h"

GLWidgetSet::GLWidgetSet(uint aNb, QColor color1, QColor color2, bool modePt) :
    _widgets(aNb),
    _zoomWidget(NULL)
{
    if (aNb==0)
        return;

    _widgets[0] = new GLWidget(0, NULL);

    for (uint aK=1 ; aK < aNb; ++aK)
        _widgets[aK] = new GLWidget( aK, (const QGLWidget*)_widgets[0]);

    QString style = "margin: 0px;"
                    "padding: 0px;";

    for (uint aK=0 ; aK < aNb; ++aK)
    {
        _widgets[aK]->setBackgroundColors(color1,color2);
        _widgets[aK]->setStyleSheet(style);
        if (!modePt) _widgets[aK]->setContextMenuPolicy( Qt::NoContextMenu );
    }

    if (modePt)
    {
        _zoomWidget = new GLWidget(-1, (const QGLWidget*)_widgets[0]);
        _zoomWidget->setBackgroundColors(color1,color1);
        _zoomWidget->setContextMenuPolicy( Qt::NoContextMenu );
        _zoomWidget->setOption(cGLData::OpShow_Mess,false);
        _zoomWidget->setZoom(3.f);
    }
}

GLWidgetSet::~GLWidgetSet()
{
    for (int aK=0; aK < nbWidgets();++aK) delete _widgets[aK];
    delete _zoomWidget;
}
