                <div class="row">
                  <div class="col-xs-4">
                    <div class="box box-standard">
                      <div class="box-header with-border">
                        <h3 class="box-title">Experiment Configuration</h3>
                      </div>
                      <div class="box-body">
                        <table class="table table-bordered">
                          %for entry in conf:
                          </tr>
                            <th>{{entry[0]}}</th>
                            <td>{{entry[1]}}</td>
                          </tr>
                          %end
                        </table>
                      </div>
                    </div>
                  </div>

                  <div class="col-xs-4">
                    %for i, graphs in enumerate(lime):
                    <div class="box box-standard">
                      <div class="box-header with-border">
                        <h3 class="box-title">LIME Evaluation {{i}}</h3>
                      </div>
                      <div class="box-body">
                        <div class="chart">
                        %for i, fl in enumerate(graphs):
                        %if i != 2:
                          <div class="col-xs-6">
                            <iframe width="100%" height="250px" src="{{fl}}" scrolling="no" frameborder="0"></iframe>
                          </div>
                        %else:
                        </div>
                        <div>
                            {{!fl}}
                        </div>
                        %end
                        %end
                      </div>
                    </div>
                    %end
                  </div>

                  <div class="col-xs-4">
                    <div class="box box-standard">
                      <div class="box-header with-border">
                        <h3 class="box-title">Performance</h3>
                      </div>
                      <div class="box-body">
                        <div class="chart">
                          <iframe width="100%" height="400px" src="{{plot}}" scrolling="no" frameborder="0"></iframe>
                        </div>
                      </div>
                    </div>
                    <div class="box box-standard">
                      <div class="box-header with-border">
                        <h3 class="box-title">Confusion Matrix</h3>
                      </div>
                      <div class="box-body">
                        <div class="chart">
                          %for t, url in heat:
                          <div class="col-xs-6">
                            <h4>{{t}}</h4>
                            <iframe width="100%" height="300px" src="{{url}}" scrolling="no" frameborder="0"></iframe>
                          </div>
                          %end
                        </div>
                      </div>
                    </div>
                  </div>
                </div>
