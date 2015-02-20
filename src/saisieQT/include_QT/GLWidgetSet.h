#ifndef GLWIDGETGRID_H
#define GLWIDGETGRID_H

#include "GLWidget.h"

class GLWidget;

#define CURRENT_IDW -1
#define THISWIN     -2

class GLWidgetSet
{
public:
    GLWidgetSet();
    ~GLWidgetSet();

    void init(cParameters *params, bool modePt);

    void setCurrentWidgetIdx(int aK);
    int  currentWidgetIdx()
    {
        return _widgets.indexOf(_pcurrentWidget);
    }

    void setCurrentWidget(GLWidget* currentWidget)
    {
        _pcurrentWidget = currentWidget;
    }

    GLWidget* getWidget(int aK = CURRENT_IDW){return aK==CURRENT_IDW ? currentWidget() :_widgets[aK];}

	GLWidget* currentWidget();

	int nbWidgets() const;

	GLWidget* zoomWidget();

    GLWidget * threeDWidget() const;

	void widgetSetResize(int);

    void option3DPreview();

    void init3DPreview(cData *data, cParameters params);

    void selectCameraIn3DP(int idCam);

private:

    QVector <GLWidget*> _widgets;
    GLWidget*           _zoomWidget;
    GLWidget*           _3DWidget;
    GLWidget*           _pcurrentWidget;

};

#endif // GLWIDGETGRID_H
