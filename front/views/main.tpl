<!DOCTYPE html>
<html>
  <head>
    <meta charset="utf-8">
    <meta http-equiv="X-UA-Compatible" content="IE=edge">
    <title>Omesa - {{ !page }}</title>
    <meta content="width=device-width, initial-scale=1, maximum-scale=1, user-scalable=no" name="viewport">
    <link rel="stylesheet" href="/static/bootstrap/css/bootstrap.min.css">
    <link rel="stylesheet" href="https://cdnjs.cloudflare.com/ajax/libs/font-awesome/4.5.0/css/font-awesome.min.css">
    <link rel="stylesheet" href="/static/dist/css/AdminLTE.css">
    <link rel="stylesheet" href="/static/dist/css/skins/skin-custom.css">
    {{ !header }}
    <!--[if lt IE 9]>
    <script src="https://oss.maxcdn.com/html5shiv/3.7.3/html5shiv.min.js"></script>
    <script src="https://oss.maxcdn.com/respond/1.4.2/respond.min.js"></script>
    <![endif]-->
  </head>
  <body class="hold-transition skin-custom sidebar-mini">
  <div class="wrapper">
    <header class="main-header">
      <a href="/" class="logo">
          <span class="logo-mini"><object height="40mm" type="image/svg+xml" data="static/omesa_plain_white.svg">O</object></span>
          <span class="logo-lg"><object height="40mm" type="image/svg+xml" data="static/omesa_plain_white.svg">O</object></span>
      </a>
      <nav class="navbar navbar-static-top">
        <a href="#" class="sidebar-toggle" data-toggle="offcanvas" role="button">
          <span class="sr-only">Toggle navigation</span>
        </a>
      </nav>
    </header>
    <aside class="main-sidebar">
      <section class="sidebar">
        <ul class="sidebar-menu">
          <li {{! 'class="active"' if page == 'Dashboard' else ''}}><a href="/"><i class="fa fa-dashboard"></i> <span>Dashboard</span></a></li>
          <li {{! 'class="active"' if page == 'Experiment' else ''}}><a href="run"><i class="fa fa-flask"></i> <span>Experiment</span></a></li>
          <li {{! 'class="active"' if page == 'Results' else ''}}><a href="exp"><i class="fa fa-line-chart"></i> <span>Results</span></a></li>
          <li {{! 'class="active"' if page == 'Data' else ''}}><a href="data"><i class="fa fa-database"></i> <span>Data</span></a></li>
          <li {{! 'class="active"' if page == 'Documentation' else ''}}><a href="docs"><i class="fa fa-book"></i> <span>Documentation</span></a></li>
        </ul>
      </section>
    </aside>
    <div class="content-wrapper">
      <section class="content-header">
        <h1>&nbsp;</h1>
        <ol class="breadcrumb">
          <li><a href="#"><i class="fa fa-dashboard"></i> Home</a></li>
          <li class="active">{{ !page }}</li>
        </ol>
      </section>
      <section class="content">
        {{ !content }}
      </section>
    </div>
    <footer class="main-footer">
      <div class="pull-right hidden-xs">
        <b>Version</b> 0.2.9a0
      </div>
      <strong>
        <p>Made by <a href="https://github.com/cmry">Chris Emmery</a>.
          Built with <a href="http://www.getbootstrap.com">Bootstrap</a>,
          <a href="https://blitzdb.readthedocs.org/en/latest/">Blitz-DB</a>,
          <a href="http://www.bottlepy.org">Bottle</a>,
          and <a href="http://www.scikit-learn.org">scikit-learn</a>.
          Themed by <a href="https://github.com/almasaeed2010/AdminLTE">AdminLTE</a>.
          Icons by <a href="http://www.fontawesome.io">Font Awesome</a>.
        </p>
      </strong>
    </footer>
  </div>
  <script src="/static/plugins/jQuery/jQuery-2.2.0.min.js"></script>
  <script src="/static/bootstrap/js/bootstrap.min.js"></script>
  <script src="/static/dist/js/app.min.js"></script>
  {{ !footer }}
  </body>
</html>
