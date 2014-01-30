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

//    QString style = "border: 2px solid #707070;"
//            "border-radius: 0px;"
//            "padding: 2px;"
//            "background: qlineargradient(x1:0, y1:0, x2:0, y2:1,stop:0 rgb(%1,%2,%3), stop:1 rgb(%4,%5,%6));";


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
    }
}

GLWidgetSet::~GLWidgetSet()
{
    for (int aK=0; aK < nbWidgets();++aK) delete _widgets[aK];
    delete _zoomWidget;
}
