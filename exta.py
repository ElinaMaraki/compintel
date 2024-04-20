# for index, element in enumerate(dicts):
#     print(f"{index}: {element}")
#
# non_zero = vectorizer.idf_
# stop_words = vectorizer.stop_words_
# print(vectorizer.get_feature_names_out())


    # print(f"Fold {idx + 1} MSE: {score:.4f}")

    # # Predict method of linear regression after training
    # predicted = model.predict(X_it_val)
    # custom_error(y_it_val, predicted)

# # Calculate and display average cross-validation score
# average_score = np.mean(scores)
# print(f"Average MSE: {average_score:.4f}")

# # Test
# test_prediction = model.predict(X_test)
# test_score = mean_squared_error(y_test, test_prediction)

# print(y_train[0, 1], y_it_train[0, 0])

# for x_sample, y_sample in zip(X_it_train, y_it_train):
#     # print(len(y_sample), y_sample)
#     model.fit(x_sample, y_sample)
# prediction = model.predict(np.expand_dims(x_sample, axis=0))
# error_sample = custom_error(y_sample, prediction)

# for x_sample, y_sample in zip(X_it_val, y_it_val):
#     prediction_val = model.predict(np.expand_dims(x_sample, axis=0))  # Predict for a single sample
#     error_val = custom_error(y_sample, prediction_val)
