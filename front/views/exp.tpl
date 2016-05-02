                <div class="row">
                    <div class="col-xs-12">
                        <div class="box">
                            <div class="box-header">
                                <h3 class="box-title">Experimental Results</h3>
                            </div>
                            <div class="box-body">
                                <table id="example1" class="table table-bordered table-striped">
                                    <thead>
                                        <tr>
                                            <th>Project</th>
                                            <th>Name</th>
                                            <th>Training Set</th>
                                            <th>Test Set</th>
                                            <th>Features</th>
                                            <th>Classifier</th>
                                            <th>Time</th>
                                            <th>Score</th>
                                        </tr>
                                    </thead>
                                    <tbody>
                                        % for id, exp in data.items():
                                        <tr>
                                            <td>{{exp['project']}}
                                            <td><a href="./exp/{{exp['name']}}">{{exp['name']}}</a></td>
                                            <td>{{exp['train_data']}}</td>
                                            <td>{{exp['test_data']}}</td>
                                            <td>{{exp['features']}}</td>
                                            <td>{{exp['clf']}}</td>
                                            <td>{{exp['dur']}}</td>
                                            <td>{{exp['test_score']}}</td>
                                        </tr>
                                        % end
                                    </tbody>
                                </table>
                            </div>
                        </div>
                    </div>
                </div>
