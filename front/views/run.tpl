<form role="form">
               <div class="row">
                   <div class="col-md-6">
                       <!-- general form elements -->
                       <div class="box box-default">
                           <div class="box-header with-border">
                               <h3 class="box-title">Info</h3>
                           </div>
                           <!-- /.box-header -->
                           <!-- form start -->
                           <div class="box-body">
                           </div>
                       </div>
                   </div>
                   <div class="col-md-6">
                       <div class="box box-default">
                           <div class="box-header with-border">
                               <h3 class="box-title">Features</h3>
                           </div>
                           <div class="box-body">
                               <div class="form-group">
                                   <textarea class="form-control" rows="5" placeholder="Ngrams(level='token', n_list=[1,2]), \n APISent(mode='deep')"></textarea>
                               </div>
                           </div>
                       </div>
                   </div>
               </div>
               <div class="row">
                   <div class="col-md-6">
                       <!-- general form elements -->
                       <div class="box box-default">
                           <div class="box-header with-border">
                               <h3 class="box-title">Data</h3>
                           </div>
                           <!-- /.box-header -->
                           <!-- form start -->
                           <div class="box-body">
                               <div class="form-group">
                                   <label>Train Data</label>
                                   <select class="form-control select2" multiple="multiple" data-placeholder="Select data" style="width: 100%;">
                                       <option>dataset_1</option>
                                       <option>dataset_train</option>
                                   </select>
                               </div>
                               <div class="form-group">
                                   <label>Test Data</label>
                                   <select class="form-control select2" multiple="multiple" data-placeholder="Select data" style="width: 100%;">
                                       <option>dataset_1</option>
                                       <option>dataset_test</option>
                                   </select>
                               </div>
                               <div class="form-group">
                                   <label>Label Selection</label>
                                   <input class="form-control" placeholder="{0: -1, 1: 50}" type="text">
                               </div>
                           </div>
                       </div>
                       <div class="box-footer">
                           <button type="submit" class="btn btn-primary">Submit</button>
                       </div>
                       <!-- /.box -->
                   </div>
                   <div class="col-md-6">
                       <div class="box box-default">
                           <div class="box-header with-border">
                               <h3 class="box-title">Classifiers</h3>
                           </div>
                           <div class="box-body">
                               <div class="form-group">
                                   <textarea class="form-control" rows="10" placeholder="Ngrams(level='token', n_list=[1,2]), \n APISent(mode='deep')"></textarea>
                               </div>
                           </div>
                       </div>
                   </div>
               </div>
           </form>
