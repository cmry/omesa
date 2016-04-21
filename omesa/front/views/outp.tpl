                <div class="row">
                    <div class="col-xs-12">
                        <div class="box">
                            <div class="box-header">
                                <h3 class="box-title">Experimental Results {{ !data }}</h3>
                            </div>
                            <div class="box-body">
                                <table id="example1" class="table table-bordered table-striped">
                                    <thead>
                                        <tr>
                                            <th>Name</th>
                                            <th>Training Set</th>
                                            <th>Test Set</th>
                                            <th>Features</th>
                                            <th>Classifier</th>
                                            <th>Result</th>
                                        </tr>
                                    </thead>
                                    <tbody>
                                        <tr>
                                            <td><a href="./exp?=test">test</a></td>
                                            <td>n_grams.csv</td>
                                            <td>n_grams.csv</td>
                                            <td>NGrams(n_list=[1, 2], level='token'), NGrams(n_list=[3], level='char'), APISent()</td>
                                            <td>LinearSVC</td>
                                            <td>0: 0.95, 1: 0.90</td>
                                        </tr>
                                        <tr>
                                            <td><a href="./exp?=test_2">test_2</a></td>
                                            <td>n_grams.csv</td>
                                            <td>n_grams.csv</td>
                                            <td>NGrams(n_list=[1], level='token'), NGrams(n_list=[3], level='char')</td>
                                            <td>LinearSVC</td>
                                            <td>0: 0.85, 1: 0.75</td>
                                        </tr>
                                    </tbody>
                                </table>
                            </div>
                        </div>
                    </div>
                </div>
